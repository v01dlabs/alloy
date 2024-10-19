//! Type checker for Alloy
//!
//! This module is responsible for performing semantic analysis and type checking
//! on the Abstract Syntax Tree (AST) produced by the parser. It ensures that
//! the program is well-typed according to Alloy's type system.

use thin_vec::ThinVec;

use crate::parser::{AstNode, BinaryOperator, TypeAnnotation, UnaryOperator};
use std::collections::HashMap;

/// Represents a type in the Alloy type system.
#[derive(Debug, PartialEq, Clone)]
pub enum Type {
    Int,
    Float,
    String,
    Bool,
    Array(Box<Type>),
    Function(ThinVec<Type>, Box<Type>),
    Void,
    Unknown,
}

/// Represents a typing error.
#[derive(Debug)]
pub struct TypeError {
    pub message: String,
}

impl std::fmt::Display for TypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Type Error: {}", self.message)
    }
}

/// The type environment stores variable and function types.
type TypeEnv = HashMap<String, Type>;

/// The main type checker struct.
pub struct TypeChecker {
    env: TypeEnv,
}

impl TypeChecker {
    /// Creates a new TypeChecker instance.
    pub fn new() -> Self {
        TypeChecker {
            env: HashMap::new(),
        }
    }

    /// Type checks an entire program.
    pub fn typecheck_program(&mut self, program: &AstNode) -> Result<Type, TypeError> {
        match program {
            AstNode::Program(statements) => {
                for stmt in statements {
                    self.typecheck_statement(stmt)?;
                }
                Ok(Type::Void)
            }
            _ => Err(TypeError {
                message: "Expected a program".to_string(),
            }),
        }
    }

    /// Type checks a statement.
    fn typecheck_statement(&mut self, stmt: &AstNode) -> Result<Type, TypeError> {
        match stmt {
            AstNode::VariableDeclaration {
                name,
                mutable: _,
                type_annotation,
                initializer,
            } => {
                let var_type = if let Some(type_ann) = type_annotation {
                    self.annotation_to_type(type_ann)
                } else if let Some(init) = initializer {
                    self.typecheck_expression(init)?
                } else {
                    return Err(TypeError {
                        message: "Cannot infer type of variable without initializer".to_string(),
                    });
                };
                self.env.insert(name.clone(), var_type.clone());
                Ok(var_type)
            }
            AstNode::FunctionDeclaration {
                name,
                generic_params: _,
                params,
                return_type,
                body,
            } => {
                let param_types: ThinVec<Type> = params
                    .iter()
                    .map(|(_, type_ann)| self.annotation_to_type(type_ann))
                    .collect();
                let return_type = return_type
                    .as_ref()
                    .map(|rt| self.annotation_to_type(rt))
                    .unwrap_or(Type::Void);
                let func_type = Type::Function(param_types, Box::new(return_type.clone()));
                self.env.insert(name.clone(), func_type);

                // Create a new scope for the function body
                let mut func_checker = TypeChecker::new();
                func_checker.env = self.env.clone();
                for (param_name, param_type) in params {
                    func_checker
                        .env
                        .insert(param_name.clone(), self.annotation_to_type(param_type));
                }

                for stmt in body {
                    let stmt_type = func_checker.typecheck_statement(stmt)?;
                    if stmt_type != return_type {
                        return Err(TypeError {
                            message: format!(
                                "Function body statement type {:?} does not match declared return type {:?}",
                                stmt_type, return_type
                            ),
                        });
                    }
                }

                Ok(Type::Void)
            }
            AstNode::ForInLoop {
                item,
                iterable,
                body,
            } => {
                let iterable_type = self.typecheck_expression(iterable)?;
                match iterable_type {
                    Type::Array(element_type) => {
                        self.env.insert(item.clone(), *element_type);
                        self.typecheck_statement(body)?;
                        Ok(Type::Void)
                    }
                    _ => Err(TypeError {
                        message: "For loop iterable must be an array".to_string(),
                    }),
                }
            }
            AstNode::WhileLoop { condition, body } => {
                let cond_type = self.typecheck_expression(condition)?;
                if cond_type != Type::Bool {
                    return Err(TypeError {
                        message: "While loop condition must be a boolean".to_string(),
                    });
                }
                self.typecheck_statement(body)?;
                Ok(Type::Void)
            }
            AstNode::Block(statements) => {
                let mut block_type = Type::Void;
                for stmt in statements {
                    block_type = self.typecheck_statement(stmt)?;
                }
                Ok(block_type)
            }
            AstNode::ReturnStatement(expr) => {
                if let Some(e) = expr {
                    self.typecheck_expression(e)
                } else {
                    Ok(Type::Void)
                }
            }
            AstNode::IfStatement {
                condition,
                then_branch,
                else_branch,
            } => {
                let cond_type = self.typecheck_expression(condition)?;
                if cond_type != Type::Bool {
                    return Err(TypeError {
                        message: "If condition must be a boolean".to_string(),
                    });
                }
                let then_type = self.typecheck_statement(then_branch)?;
                if let Some(else_branch) = else_branch {
                    let else_type = self.typecheck_statement(else_branch)?;
                    if then_type != else_type {
                        return Err(TypeError {
                            message: "If and else branches must have the same type".to_string(),
                        });
                    }
                }
                Ok(then_type)
            }
            _ => self.typecheck_expression(stmt),
        }
    }

    /// Type checks an expression.
    fn typecheck_expression(&self, expr: &AstNode) -> Result<Type, TypeError> {
        match expr {
            AstNode::IntLiteral(_) => Ok(Type::Int),
            AstNode::FloatLiteral(_) => Ok(Type::Float),
            AstNode::StringLiteral(_) => Ok(Type::String),
            AstNode::BoolLiteral(_) => Ok(Type::Bool),
            AstNode::Identifier(name) => self.env.get(name).cloned().ok_or(TypeError {
                message: format!("Undefined variable: {}", name),
            }),
            AstNode::ArrayLiteral(elements) => self.typecheck_array_literal(elements),
            AstNode::BinaryOperation {
                left,
                operator,
                right,
            } => {
                let left_type = self.typecheck_expression(left)?;
                let right_type = self.typecheck_expression(right)?;
                self.typecheck_binary_op(operator, &left_type, &right_type)
            }
            AstNode::UnaryOperation { operator, operand } => {
                let operand_type = self.typecheck_expression(operand)?;
                self.typecheck_unary_op(operator, &operand_type)
            }
            AstNode::FunctionCall { arguments, callee } => {
                let func_type = self.typecheck_expression(callee)?;
                match func_type {
                    Type::Function(param_types, return_type) => {
                        if arguments.len() != param_types.len() {
                            return Err(TypeError {
                                message: "Wrong number of arguments".to_string(),
                            });
                        }
                        for (arg, expected_type) in arguments.iter().zip(param_types.iter()) {
                            let arg_type = self.typecheck_expression(arg)?;
                            if arg_type != *expected_type {
                                return Err(TypeError {
                                    message: format!(
                                        "Argument type mismatch: expected {:?}, got {:?}",
                                        expected_type, arg_type
                                    ),
                                });
                            }
                        }
                        Ok(*return_type)
                    }
                    _ => Err(TypeError {
                        message: "Calling non-function type".to_string(),
                    }),
                }
            }
            _ => Err(TypeError {
                message: "Unexpected node in expression".to_string(),
            }),
        }
    }

    /// Checks the types for a binary operation.
    fn typecheck_binary_op(
        &self,
        op: &BinaryOperator,
        left: &Type,
        right: &Type,
    ) -> Result<Type, TypeError> {
        match op {
            BinaryOperator::Add
            | BinaryOperator::Subtract
            | BinaryOperator::Multiply
            | BinaryOperator::Divide => {
                if left == right && (left == &Type::Int || left == &Type::Float) {
                    Ok(left.clone())
                } else {
                    Err(TypeError {
                        message: "Invalid types for arithmetic operation".to_string(),
                    })
                }
            }
            BinaryOperator::Equal | BinaryOperator::NotEqual => {
                if left == right {
                    Ok(Type::Bool)
                } else {
                    Err(TypeError {
                        message: "Incompatible types for comparison".to_string(),
                    })
                }
            }
            BinaryOperator::LessThan
            | BinaryOperator::GreaterThan
            | BinaryOperator::LessThanOrEqual
            | BinaryOperator::GreaterThanOrEqual => {
                if left == right && (left == &Type::Int || left == &Type::Float) {
                    Ok(Type::Bool)
                } else {
                    Err(TypeError {
                        message: "Invalid types for comparison".to_string(),
                    })
                }
            }
            BinaryOperator::And | BinaryOperator::Or => {
                if left == &Type::Bool && right == &Type::Bool {
                    Ok(Type::Bool)
                } else {
                    Err(TypeError {
                        message: "Boolean operation requires boolean operands".to_string(),
                    })
                }
            }
            BinaryOperator::Assign => todo!(),
            BinaryOperator::Pipeline => todo!(),
        }
    }

    /// Checks the types for a unary operation.
    fn typecheck_unary_op(&self, op: &UnaryOperator, operand: &Type) -> Result<Type, TypeError> {
        match op {
            UnaryOperator::Negate => {
                if operand == &Type::Int || operand == &Type::Float {
                    Ok(operand.clone())
                } else {
                    Err(TypeError {
                        message: "Negation requires numeric operand".to_string(),
                    })
                }
            }
            UnaryOperator::Not => {
                if operand == &Type::Bool {
                    Ok(Type::Bool)
                } else {
                    Err(TypeError {
                        message: "Logical NOT requires boolean operand".to_string(),
                    })
                }
            }
            UnaryOperator::Increment => todo!(),
        }
    }

    /// Checks the type of an array literal.
    fn typecheck_array_literal(&self, elements: &[AstNode]) -> Result<Type, TypeError> {
        if elements.is_empty() {
            return Ok(Type::Array(Box::new(Type::Unknown)));
        }

        let first_type = self.typecheck_expression(&elements[0])?;
        for element in elements.iter().skip(1) {
            let element_type = self.typecheck_expression(element)?;
            if element_type != first_type {
                return Err(TypeError {
                    message: format!(
                        "Inconsistent types in array literal: {:?} and {:?}",
                        first_type, element_type
                    ),
                });
            }
        }

        Ok(Type::Array(Box::new(first_type)))
    }

    /// Converts a TypeAnnotation to a Type.
    fn annotation_to_type(&self, annotation: &TypeAnnotation) -> Type {
        match annotation {
            TypeAnnotation::Int => Type::Int,
            TypeAnnotation::Float => Type::Float,
            TypeAnnotation::String => Type::String,
            TypeAnnotation::Bool => Type::Bool,
            TypeAnnotation::Array(inner_type) => {
                Type::Array(Box::new(self.annotation_to_type(inner_type)))
            }
            TypeAnnotation::Custom(name) => self.env.get(name).cloned().unwrap_or(Type::Unknown),
            TypeAnnotation::Simple(_) => todo!(),
            TypeAnnotation::Generic(_, vec) => todo!(),
            TypeAnnotation::Function(vec, type_annotation) => todo!(),
        }
    }
}

/// Type checks an AST node.
pub fn typecheck(ast: &AstNode) -> Result<(), TypeError> {
    let mut checker = TypeChecker::new();
    checker.typecheck_program(ast)?;
    Ok(())
}
