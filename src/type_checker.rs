

//! Type checker for Alloy
//!
//! This module is responsible for performing semantic analysis and type checking
//! on the Abstract Syntax Tree (AST) produced by the parser. It ensures that
//! the program is well-typed according to Alloy's type system.

use thin_vec::{thin_vec, ThinVec};
use thiserror::Error;

use crate::{
    
    ast::{AstNode, BinaryOperator, UnaryOperator, P}, 
    ty::{AttrItem, BindingMode, Const, FnRetTy, Ident, IntTy,PatField, Path, Pattern, PatternKind, RefKind, Ty, TyKind, TypeOp}, 
};

use std::collections::HashMap;



/// Represents a typing error.
#[derive(Debug, Error)]
pub struct TypeError {
    pub message: String,
}

impl std::fmt::Display for TypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Type Error: {}", self.message)
    }
}


/// The type environment stores variable and function types.
type TypeEnv = HashMap<String, Box<Type>>;

/// The main type checker struct.
pub struct TypeChecker {
    env: TypeEnv,
    name_index: usize
}

impl TypeChecker {
    /// Creates a new TypeChecker instance.
    pub fn new() -> Self {
        TypeChecker {
            env: HashMap::new(),
            name_index: 0
        }
    }

    pub fn new_name(&mut self) -> String {
        let new_index = self.name_index + 1;
        self.name_index = new_index;
        format!("{}", new_index)
    }

    pub fn add_anon_type(&mut self, ty: &Type) {
        let new_name = self.new_name();
        self.env.insert(new_name, P(ty.clone()));
    }

    pub fn copy_env(&self) -> Self {
        TypeChecker {
            env: self.env.clone(),
            name_index: self.name_index
        }
    }

    pub fn resolve_ident(&self, ident: &Ident) -> Result<Box<Type>, TypeError> {
        match self.env.get(ident) {
            Some(ty) => Ok(P(*ty.clone())),
            None => Err(TypeError { message: format!("Undefined variable {}", ident) })
        }
    }

    pub fn resolve_function(&self, name: &Ident) -> Result<Box<Type>, TypeError> {
        match self.env.get(name) {
            Some(ty) => match ty.clone() {
                box Type::Function(_) => Ok(P(*ty.clone())),
                _ => Err(TypeError { message: format!("Expected function, got {:?}", ty) })
            },   
            None => Err(TypeError { message: format!("Undefined function {}", name) })
        }
    }

    pub fn infer_type(&mut self, ast: &AstNode) -> Result<Type, TypeError> {
        match ast {
            AstNode::Program(program) => {
                let mut last_type = Type::Infer;
                for statement in program.iter() {
                    last_type = self.infer_type(statement)?;
                }
                Ok(last_type)
            },
            AstNode::FunctionDeclaration { 
                name, 
                function: crate::ty::Function { generic_params, inputs, output }, 
                body } => {
                    let mut fn_checker = self.copy_env();
                    for generic_param in generic_params.iter() {
                        match &generic_param.kind {
                            crate::ty::GenericParamKind::Type(Some( box ty)) => {
                                let ty = P(Type::from(ty.clone()));
                                fn_checker.env.insert(generic_param.name.clone(), ty.clone());
                            },
                            crate::ty::GenericParamKind::Type(None) => {
                                fn_checker.env.insert(generic_param.name.clone(), P(Type::Infer));
                            },
                            crate::ty::GenericParamKind::Const { box ty, .. } => {
                                let ty = P(Type::from(ty.clone()));
                                fn_checker.env.insert(generic_param.name.clone(), ty);
                            },
                        }
                    }
                    let param_types: ThinVec<_> = inputs
                        .iter()
                        .map(|param| {
                            let ty = P(Type::from(*param.ty.clone()));
                            fn_checker.env.insert(param.name.clone(), ty.clone());
                            ty
                        })
                        .collect();
                    let mut return_type = Type::from(output.clone());
                    for statement in body.iter() {
                        if let &box AstNode::ReturnStatement(ref expr) = statement {
                            let statement_type = if let Some(expr) = expr {
                                fn_checker.infer_type(expr)?
                            } else {
                                Type::unit()
                            };
                            if statement_type.infer_or_type(&return_type) != return_type.infer_or_type(&statement_type) {
                                return Err(TypeError { 
                                    message: format!("return statement has type: {:?} but function specifies type: {:?}", 
                                        statement_type, return_type)
                                });
                            }
                            return_type = statement_type.infer_or_type(&return_type).clone();
                            break;
                        }
                        let statement_type = fn_checker.infer_type(statement)?;
                        fn_checker.add_anon_type(&statement_type);
                    }
                    let fn_type = Type::Function(Function { 
                        generic_params: generic_params.clone().into_iter().map(Into::into).collect(),
                        inputs: inputs.iter().zip(param_types.iter()).map(|(t1, t2)| {
                            Param {
                                name: t1.name.clone(),
                                ty: P(*t2.clone())
                            }
                        }).collect(), 
                        output: P(return_type)
                    });
                    self.env.insert(name.clone(), P(fn_type.clone()));
                    Ok(fn_type)
                },
            AstNode::VariableDeclaration { 
                name, mutable, type_annotation, initializer 
            } => {
                // TODO: handle mutability
                let var_type = annotation_to_type(type_annotation);
                if let Some(initializer) = initializer {
                    if var_type == Type::Infer {
                        let inferred_type = self.infer_type(initializer)?;
                        self.env.insert(name.clone(), P(inferred_type.clone()));
                        Ok(inferred_type)
                    } else {
                        let inferred_type = self.infer_type(initializer)?;
                        if inferred_type != Type::Infer && inferred_type != var_type {
                            Err(TypeError {
                                message: format!("Type mismatch: expected {:?}, got {:?}", var_type, inferred_type),
                            })
                        } else if inferred_type == var_type {
                            self.env.insert(name.clone(), P(var_type.clone()));
                            Ok(var_type)
                        } else {
                            // currently an error, but we can likely recover by inferring the type from other usages
                            Err(TypeError {
                                message: format!("Couldn't infer type for variable {} from initializer {:?}", name, initializer),
                            })
                        }
                    }
                } else {
                    self.env.insert(name.clone(), P(var_type.clone()));
                    Ok(var_type)
                }
            },
            AstNode::IfStatement { condition, then_branch, else_branch } => {
                let cond_type = self.infer_type(&condition)?;
                if cond_type != Type::Bool {
                    Err(TypeError { message: format!("conditional expression `{:?}` should resolve to a bool", cond_type) })
                } else {
                    let then_type = self.infer_type(then_branch)?;
                    if let Some(else_branch) = else_branch {
                        let else_type = self.infer_type(else_branch)?;
                        if then_type != else_type {
                            Err(TypeError { 
                                message: format!("then and else branches of if statement should have the same type, 
                                    but they have different types: {:?} and {:?}", then_type, else_type) })
                        } else {
                            Ok(then_type)
                        }
                    } else {
                        Ok(then_type)
                    }
                }
            },
            AstNode::WhileLoop { condition, body } => {
                let cond_type = self.infer_type(&condition)?;
                if cond_type != Type::Bool {
                    Err(TypeError { message: format!("conditional expression `{:?}` should resolve to a bool", cond_type) })
                } else {
                    self.infer_type(body)
                }
            },
            AstNode::ForInLoop { item, iterable, body } => {
                let iter_type = self.infer_type(iterable)?;
                if let Type::Array(ty) = iter_type {
                    let mut loop_checker = self.copy_env();
                    loop_checker.env.insert(item.clone(), ty.clone());
                    loop_checker.infer_type(body)
                } else {
                    // For now until we have other things to iterate over
                    Err(TypeError { message: format!("iterable `{:?}` should resolve to an array", iter_type) })
                }
            },
            AstNode::GuardStatement { condition, body } => {
                let cond_type = self.infer_type(&condition)?;
                if cond_type != Type::Bool {
                    Err(TypeError { message: format!("conditional expression `{:?}` should resolve to a bool", cond_type) })
                } else {
                    let body_type = self.infer_type(body)?;
                    Ok(body_type)
                }
            },
            AstNode::ReturnStatement(ast_node) => {
                if let Some(expr) = ast_node {
                    self.infer_type(expr)
                } else {
                    Ok(Type::unit())
                }
            },
            AstNode::Block(block) => {
                let mut last_type = Type::Infer;
                for statement in block.iter() {
                    last_type = self.infer_type(statement)?;
                }
                Ok(last_type)
            },
            AstNode::BinaryOperation { left, operator, right } => {
                let left_type = self.infer_type(left)?;
                let right_type = self.infer_type(right)?;
                self.typecheck_binary_op(operator, &left_type, &right_type)
            },
            AstNode::UnaryOperation { operator, operand } => {
                let operand_type = self.infer_type(operand)?;
                self.typecheck_unary_op(operator, &operand_type)
            },
            AstNode::FunctionCall { callee, arguments } => {
                let callee_type = self.infer_type(callee)?;
                if let Type::Function(function) = callee_type {
                    if arguments.len() != function.inputs.len() {
                        return Err(TypeError { 
                            message: format!("Expected {} arguments, got {}", function.inputs.len(), arguments.len()) 
                        });
                    }
                    let return_type = function.output.clone();
                    for (argument, param_type) in arguments.iter().zip(function.inputs.iter()) {
                        let arg_type = self.infer_type(argument)?;
                        if arg_type != *param_type.ty {
                            return Err(TypeError { 
                                message: format!("Argument type mismatch: expected {:?}, got {:?}", param_type, arg_type) 
                            });
                        }
                    }
                    Ok(*return_type)
                } else {
                    Err(TypeError { message: format!("Expected function, got {:?}", callee_type) })
                }
            },
            AstNode::GenericFunctionCall { name, generic_args, arguments } => {
                let callee_type = self.env.get(name).cloned()
                    .ok_or(TypeError { message: format!("Undefined function {}", name) })?;
                match *callee_type {
                    Type::Function(function) => self.typecheck_function_call(&function, arguments, generic_args),
                    e => Err(TypeError { message: format!("Expected function, got {:?}, error: {:?}", name, e) })
                }
            },
            AstNode::TrailingClosure { callee, closure } => {
                let mut full_args: ThinVec<Box<AstNode>> = thin_vec![];
                let mut generic_arguments: ThinVec<Box<Ty>> = thin_vec![];
                let callee_type = match callee {
                    &box AstNode::GenericFunctionCall { 
                        ref name, ref generic_args, ref arguments
                    } => {
                        full_args.append(&mut arguments.clone());
                        // Currently assuming the last argument is the closure, as in Kotlin
                        full_args.push(closure.clone());
                        generic_arguments.append(&mut generic_args.clone());
                        self.env.get(name).cloned()
                            .ok_or(TypeError { message: format!("Undefined function {}", name) })?
                    }
                    &box AstNode::FunctionCall {ref callee, ref arguments } => {
                        full_args.append(&mut arguments.clone());
                        // Currently assuming the last argument is the closure, as in Kotlin
                        full_args.push(closure.clone());
                        P(self.infer_type(&callee)?)
                    },
                    &box AstNode::Identifier(ref name) => {
                        self.env.get(name).cloned()
                            .ok_or(TypeError { message: format!("Undefined function {}", name) })?
                    }
                    &box ref e => return Err(TypeError { 
                        message: format!("Expected function call, got {:?}, error: {:?}", callee, e) 
                    })
                };
                match *callee_type {
                    Type::Function(function) => {
                        self.typecheck_function_call(&function, &full_args, &generic_arguments)
                    },
                    e => Err(TypeError { message: format!("Expected function, got {:?}, error: {:?}", callee, e) })
                }
            },
            AstNode::PipelineOperation { left, right } => {
                //TODO: Placeholder implementation
                let left_type = self.infer_type(left)?;
                let right_type = self.infer_type(right)?;
                //self.typecheck_binary_op(&BinaryOperator::Pipeline, &left_type, &right_type)
                Ok(right_type)
            },
            AstNode::Identifier(ident) => self.env.get(ident).cloned()
                .map(|ty| *ty).ok_or(TypeError { message: format!("Undefined variable {}", ident) }),  
            AstNode::IntLiteral(_) => Ok(Type::Int(IntTy::Int)),
            AstNode::FloatLiteral(_) => Ok(Type::Float),
            AstNode::StringLiteral(_) => Ok(Type::String),
            AstNode::BoolLiteral(_) => Ok(Type::Bool),
            AstNode::ArrayLiteral(elements) => self.typecheck_array_literal(elements),
        }
    }

    fn typecheck_function_call(&mut self, 
        function: &Function, 
        arguments: &ThinVec<Box<AstNode>>, 
        generic_args: &ThinVec<Box<Ty>>
    ) -> Result<Type, TypeError> {
        if arguments.len() != function.inputs.len() {
            return Err(TypeError { 
                message: format!("Expected {} arguments, got {}", function.inputs.len(), arguments.len()) 
            });
        }
        let generic_args: ThinVec<_> = generic_args.iter().map(|ty| P(Type::from(*ty.clone()))).collect();
        let mut generic_checker = self.copy_env();
        if generic_args.len() != function.generic_params.len() {
            // TODO: figure out if we should be able to infer generic arguments
            return Err(TypeError { 
                message: format!("Expected {} generic arguments, got {}", function.generic_params.len(), generic_args.len()) 
            });
        }
        let generic_params = function.generic_params.clone();
        for (generic_arg, generic_param) in generic_args.iter().zip(generic_params.iter()) {
            match &generic_param.kind {
                GenericParamKind::Type(Some(box ty)) => {
                    if **generic_arg != *ty {
                        return Err(TypeError { 
                            message: format!("Generic argument type mismatch: expected {:?}, got {:?}", ty, generic_arg) 
                        });
                    }
                    generic_checker.env.insert(generic_param.name.clone(), P(ty.clone()));
                },
                GenericParamKind::Type(None) => {
                    generic_checker.env.insert(generic_param.name.clone(), generic_arg.clone());
                },
                GenericParamKind::Const { box ty, value: None } => {
                    if **generic_arg != *ty {
                        return Err(TypeError { 
                            message: format!("Generic argument type mismatch: expected {:?}, got {:?}", ty, generic_arg) 
                        });
                    }
                    generic_checker.env.insert(generic_param.name.clone(), generic_arg.clone());  
                },
                GenericParamKind::Const { box ty, value: Some(value) } => {
                    if **generic_arg != *ty {
                        return Err(TypeError { 
                            message: format!("Generic argument type mismatch: expected {:?}, got {:?}", ty, generic_arg) 
                        });
                    }
                    generic_checker.env.insert(generic_param.name.clone(), P(Type::Const(value.clone())));  
                },
            }
        }
        let return_type = function.output.clone();
        for (argument, param_type) in arguments.iter().zip(function.inputs.iter()) {
            let arg_type = generic_checker.infer_type(argument)?;
            if arg_type != *param_type.ty {
                return Err(TypeError { 
                    message: format!("Argument type mismatch: expected {:?}, got {:?}", param_type, arg_type) 
                });
            }
        }
        Ok(*return_type)
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
            | BinaryOperator::Divide
            | BinaryOperator::Modulo => {
                if right == &Type::Infer && ( matches!(left, &Type::Int(_)) || left == &Type::Float) {
                    Ok(right.clone())
                } else if left == &Type::Infer && ( matches!(right, &Type::Int(_)) || right == &Type::Float) {
                    Ok(left.clone())
                } else if left == right && ( matches!(left, &Type::Int(_)) || left == &Type::Float) {
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
                if right == &Type::Infer && ( matches!(left, &Type::Int(_)) || left == &Type::Float) {
                    Ok(Type::Bool)
                } else if left == &Type::Infer && ( matches!(right, &Type::Int(_)) || right == &Type::Float) {
                    Ok(Type::Bool)
                } else if left == right && ( matches!(left, &Type::Int(_)) || left == &Type::Float) {
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
            BinaryOperator::Assign => {
                //TODO: Placeholder
                Ok(Type::unit())
            },
            BinaryOperator::Pipeline => todo!(),
        }
    }

    /// Checks the types for a unary operation.
    fn typecheck_unary_op(&self, op: &UnaryOperator, operand: &Type) -> Result<Type, TypeError> {
        match op {
            UnaryOperator::Negate => {
                if matches!(operand, &Type::Int(_)) || operand == &Type::Float {
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

    pub fn typecheck_array_literal(&mut self, elements: &ThinVec<Box<AstNode>>) -> Result<Type, TypeError> {
        if elements.is_empty() {
            return Ok(Type::Array(P(Type::Infer)));
        }
        let mut first_type = self.infer_type(&elements.first().unwrap())?;
        for element in elements.iter().skip(1) {
            let element_type = self.infer_type(&element)?;
            if first_type == Type::Infer {
                // If we couldn't immediately figure out the first type, maybe the next one will work
                first_type = element_type.clone();
            }
            if first_type != element_type {
                return Err(TypeError {
                    message: format!(
                        "Inconsistent types in array literal: {:?} and {:?}",
                        first_type, element_type
                    ),
                });
            }
        }

        Ok(Type::Array(P(first_type)))
    }

    pub fn typecheck_program(&mut self, program: &AstNode) -> Result<(), TypeError> {
        self.infer_type(program)?;
        Ok(())
    }
}

/// Type checks an AST node.
pub fn typecheck(ast: &AstNode) -> Result<(), TypeError> {
    let mut checker = TypeChecker::new();
    checker.typecheck_program(ast)?;
    Ok(())
}

pub fn annotation_to_type(annotation: &Option<Box<Ty>>) -> Type {
    match annotation {
        Some(ty) => Type::from(*ty.clone()).normalize(),
        None => Type::Infer,
    }
}


/// Represents a type in the Alloy type system.
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Int(IntTy),
    Float, 
    Bool,
    Char, 
    String,
    Array(Box<Type>),
    Path(Path),
    Tuple(ThinVec<Box<Type>>),
    SizedArray(Box<Type>, Const),
    Function(Function),
    Const(Const),
    Algebraic(AlgebraicType),
    Generic(Ident, ThinVec<Box<Type>>),
    Paren(Box<Type>),
    Ref(RefKind, Box<Type>),
    Pattern(Box<Type>, Box<PatternType>), 
    Simple(Ident),
    Any,
    Infer,
    SelfType,
    Never,
    Err,
}

impl Type {
    pub fn unit() -> Self {
        Type::Tuple(thin_vec![])
    }

    pub fn infer_or_type<'ty>(&'ty self, other: &'ty Type) -> &'ty Type {
        match (self, other) {
            (Type::Infer, _) => other,
            (_, Type::Infer) => self,
            _ => self
        }
    }

    pub fn normalize(self) -> Self {
        match self {
            Type::Simple(name) => match name.as_str() {
                "int" => Type::Int(IntTy::Int),
                "float" => Type::Float,
                "bool" => Type::Bool,
                "char" => Type::Char,
                "string" => Type::String,
                "Array" => Type::Array(P(Type::Infer)),
                _ => Type::Simple(name),
            },
            Type::Generic(name, types) => match name.as_str() {
                "Array" => {
                    let elem_type = types.first()
                        .map(|ty| ty.clone().normalize()).unwrap_or(Type::Infer);
                    Type::Array(P(elem_type))
                }
                _ => Type::Generic(name, types),
            },
            _ => self,   
        }
    }   
}

impl From<&Option<Box<Ty>>> for Type {
    fn from(ty: &Option<Box<Ty>>) -> Self {
        match ty {
            Some(ty) => Type::from(*ty.clone()),
            None => Type::Infer,
        }
    }
}

impl From<Ty> for Type {
    fn from(ty: Ty) -> Self {
        match ty.kind {
            TyKind::Int(ty) => Type::Int(ty),
            TyKind::Float => Type::Float,
            TyKind::Bool => Type::Bool,
            TyKind::Char => Type::Char,
            TyKind::String => Type::String,
            TyKind::Array(ty) => Type::Array(P(Type::from(*ty))),
            TyKind::Path(path) => Type::Path(path),
            TyKind::Tuple(types) => Type::Tuple(types.into_iter()
                .map(|ty| P(Type::from(*ty).normalize())).collect()
            ),   
            TyKind::SizedArray(ty, size) => Type::SizedArray(P(Type::from(*ty)), size),
            TyKind::Function(function) => Type::Function(function.into()),
            TyKind::Const(const_) => Type::Const(const_),   
            TyKind::Algebraic(type_op) => Type::Algebraic(type_op.into()),   
            TyKind::Generic(name, types) => {
                Type::Generic(name, types.into_iter()
                    .map(|ty| P(Type::from(*ty).normalize())).collect()
                ).normalize()
            },
            TyKind::Paren(ty) => Type::Paren(P(Type::from(*ty).normalize())),
            TyKind::Ref(kind, ty) => Type::Ref(kind, P(Type::from(*ty).normalize())),
            TyKind::Pattern(ty, pat) => {
                Type::Pattern(P(Type::from(*ty).normalize()), P(PatternType::from(*pat)))
            },
            TyKind::Simple(name) => Type::Simple(name).normalize(),
            TyKind::Any => Type::Any,
            TyKind::Infer => Type::Infer,
            TyKind::SelfType => Type::SelfType,
            TyKind::Never => Type::Never,
            TyKind::Err => Type::Err,
        }
    }
}

impl From<FnRetTy> for Type {
    fn from(fn_ret_ty: FnRetTy) -> Self {
        match fn_ret_ty {
            FnRetTy::Infer => Type::Infer,
            FnRetTy::Ty(ty) => Type::from(*ty).normalize(),
        }
    }
}



#[derive(Debug, Clone, PartialEq)] 
pub enum AlgebraicType {
    And(ThinVec<Box<Type>>),
    Or(ThinVec<Box<Type>>),
    Xor(ThinVec<Box<Type>>),
    Not(Box<Type>),
    Subset(Box<Type>),
    Implements(Box<Type>),
}

impl From<TypeOp> for AlgebraicType {
    fn from(type_op: TypeOp) -> Self {
        match type_op {
            TypeOp::And(types) => AlgebraicType::And(
                types.into_iter().map(|ty| P(Type::from(*ty).normalize())).collect()
            ),
            TypeOp::Or(types) => AlgebraicType::Or(
                types.into_iter().map(|ty| P(Type::from(*ty).normalize())).collect()
            ),
            TypeOp::Xor(types) => AlgebraicType::Xor(
                types.into_iter().map(|ty| P(Type::from(*ty).normalize())).collect()
            ),
            TypeOp::Not(ty) => AlgebraicType::Not(P(Type::from(*ty).normalize())),
            TypeOp::Subset(ty) => AlgebraicType::Subset(P(Type::from(*ty).normalize())),
            TypeOp::Implements(ty) => AlgebraicType::Implements(P(Type::from(*ty).normalize())),
        }
    }
}


#[derive(Debug, Clone, PartialEq)]
pub enum PatternType {
    Wildcard,
    Ident(BindingMode, Ident, Option<Box<PatternType>>),
    Tuple(ThinVec<Box<PatternType>>),
    TupleStruct(Option<Box<QualifiedSelf>>, Path, ThinVec<Box<PatternType>>),
    Struct(Option<Box<QualifiedSelf>>, Path, ThinVec<PatternField>),
    Path(Option<Box<QualifiedSelf>>, Path),
    Ref(Box<PatternType>, RefKind),
    Or(ThinVec<Box<PatternType>>),
    Algebraic(AlgebraicType, Box<PatternType>),
    Paren(Box<PatternType>),
    Never,
    Err,
}


impl From<Pattern> for PatternType {
    fn from(pattern: Pattern) -> Self {
        match pattern.kind {
            PatternKind::Wildcard => PatternType::Wildcard,
            PatternKind::Ident(mode, name, pat) => PatternType::Ident(
                mode, name, pat.map(|pat| P(PatternType::from(*pat)))
            ),
            PatternKind::Tuple(patterns) => PatternType::Tuple(
                patterns.into_iter().map(|pat| P(PatternType::from(*pat))).collect()
            ),
            PatternKind::TupleStruct(qual_self, path, patterns) => PatternType::TupleStruct(
                qual_self.map(|qual_self| P(QualifiedSelf::from(*qual_self))), 
                path, patterns.into_iter().map(|pat| P(PatternType::from(*pat))).collect()
            ),
            PatternKind::Struct(qual_self, path, patterns) => PatternType::Struct(
                qual_self.map(|qual_self| P(QualifiedSelf::from(*qual_self))), 
                path, patterns.into_iter().map(|pat| PatternField::from(pat)).collect()
            ),
            PatternKind::Path(qual_self, path) => PatternType::Path(
                qual_self.map(|qual_self| P(QualifiedSelf::from(*qual_self))), path
            ),
            PatternKind::Ref(pat, kind) => PatternType::Ref(P(PatternType::from(*pat)), kind),
            PatternKind::Or(patterns) => PatternType::Or(
                patterns.into_iter().map(|pat| P(PatternType::from(*pat))).collect()
            ),
            PatternKind::TypeOp(type_op, pat) => PatternType::Algebraic(
                type_op.into(), P(PatternType::from(*pat))
            ),
            PatternKind::Paren(pat) => PatternType::Paren(P(PatternType::from(*pat))),
            PatternKind::Never => PatternType::Never,
            PatternKind::Err => PatternType::Err,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct QualifiedSelf {
    pub ty: Box<Type>,
}

impl From<crate::ty::QualifiedSelf> for QualifiedSelf {
    fn from(qual_self: crate::ty::QualifiedSelf) -> Self {
        QualifiedSelf {
            ty: P(Type::from(*qual_self.ty).normalize()),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PatternField {
    /// The identifier for the field.
    pub ident: Ident,
    /// The pattern the field is destructured to.
    pub pat: Box<PatternType>,
    pub is_shorthand: bool,
    pub attrs: ThinVec<AttrItem>,
    pub is_placeholder: bool,
}

impl From<PatField> for PatternField {
    fn from(pat_field: PatField) -> Self {
        PatternField {
            ident: pat_field.ident,
            pat: P(PatternType::from(*pat_field.pat)),
            is_shorthand: pat_field.is_shorthand,
            attrs: pat_field.attrs,
            is_placeholder: pat_field.is_placeholder,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub generic_params: ThinVec<GenericParam>,
    pub inputs: ThinVec<Param>,
    pub output: Box<Type>,
}

impl From<crate::ty::Function> for Function {
    fn from(function: crate::ty::Function) -> Self {
        Self {
            generic_params: function.generic_params.into_iter().map(Into::into).collect(),
            inputs: function.inputs.into_iter().map(Into::into).collect(),
            output: P(Type::from(function.output)),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GenericParam {
    pub name: Ident,
    pub kind: GenericParamKind,
    pub attrs: ThinVec<AttrItem>,
    pub bounds: Option<AlgebraicType>,
    pub is_placeholder: bool,
}

impl GenericParam {
    pub fn simple(name: Ident) -> Self {
        GenericParam {
            name,
            kind: GenericParamKind::Type(None),
            attrs: ThinVec::new(),
            bounds: None,
            is_placeholder: false,
        }
    }
}

impl From<crate::ty::GenericParam> for GenericParam {
    fn from(param: crate::ty::GenericParam) -> Self {
        Self {
            name: param.name,
            kind: param.kind.into(),
            attrs: param.attrs,
            bounds: param.bounds.map(AlgebraicType::from),
            is_placeholder: param.is_placeholder,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum GenericParamKind {
    Type(Option<Box<Type>>),
    Const {
        ty: Box<Type>,
        value: Option<Const>
    }
}

impl From<crate::ty::GenericParamKind> for GenericParamKind {
    fn from(kind: crate::ty::GenericParamKind) -> Self {
        match kind {
            crate::ty::GenericParamKind::Type(ty) => Self::Type(
                ty.map(|ty| { P(Type::from(*ty).normalize()) })
            ),
            crate::ty::GenericParamKind::Const { ty, value } => Self::Const { 
                ty: P(Type::from(*ty).normalize()), value 
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: Ident,
    pub ty: Box<Type>,
} 

impl From<crate::ty::Param> for Param {
    fn from(param: crate::ty::Param) -> Self {
        Param {
            name: param.name,
            ty: P(Type::from(*param.ty).normalize())
        }
    }
}