//! Type checker for Alloy
//!
//! This module is responsible for performing semantic analysis and type checking
//! on the Abstract Syntax Tree (AST) produced by the parser. It ensures that
//! the program is well-typed according to Alloy's type system.

use thin_vec::{thin_vec, ThinVec};
use tracing::{error, instrument};

use crate::{
    ast::{
        self,
        ty::{
            AttrItem, Const, FloatKind, FnRetTy, Ident, IntKind, Mutability, PatField, Path,
            Pattern, PatternKind, RefKind, Ty, TyKind, TypeOp, UintKind,
        },
        AstElem, AstElemKind, BinaryOperator, BindAttr, Expr, ExprKind, Item, ItemKind, Statement,
        StatementKind, UnaryOperator, Visibility, P,
    },
    error::TypeError,
};

use std::collections::HashMap;

/// The type environment stores variable and function types.
type TypeEnv = HashMap<String, Box<TypeBinding>>;

/// The main type checker struct.
#[derive(Debug)]
pub struct TypeChecker {
    env: TypeEnv,
    name_index: usize,
}

impl TypeChecker {
    /// Creates a new TypeChecker instance.
    pub fn new() -> Self {
        TypeChecker {
            env: HashMap::new(),
            name_index: 0,
        }
    }

    pub fn new_name(&mut self) -> String {
        let new_index = self.name_index + 1;
        self.name_index = new_index;
        format!("{}", new_index)
    }

    pub fn add_anon_type(&mut self, ty: &Type) {
        let new_name = self.new_name();
        self.env.insert(
            new_name,
            P(TypeBinding::variable(
                ty.clone(),
                BindAttr::new(false, None),
            )),
        );
    }

    pub fn copy_env(&self) -> Self {
        TypeChecker {
            env: self.env.clone(),
            name_index: self.name_index,
        }
    }

    pub fn insert_variable(&mut self, name: &str, ty: &Type, binding: BindAttr) {
        self.env.insert(
            name.to_string(),
            P(TypeBinding::variable(ty.clone(), binding)),
        );
    }

    pub fn insert_type(&mut self, name: &str, ty: &Type) {
        self.env.insert(
            name.to_string(),
            P(TypeBinding::new(
                ty.clone(),
                Binding {
                    visibility: Visibility::Local(None),
                    ty: BindingType::Value(BindAttr::new(false, None)),
                },
            )),
        );
    }

    pub fn insert_function(
        &mut self,
        name: &str,
        ty: &Type,
        fn_attr: FnAttr,
        visibility: Option<Visibility>,
    ) {
        self.env.insert(
            name.to_owned(),
            P(TypeBinding::function(ty.clone(), fn_attr, visibility)),
        );
    }

    pub fn resolve_ident(&self, ident: &Ident) -> Result<Box<TypeBinding>, TypeError> {
        match self.env.get(ident) {
            Some(ty) => Ok(P(*ty.clone())),
            None => Err(TypeError {
                message: format!("Undefined variable {}", ident),
            }),
        }
    }

    #[instrument]
    pub fn resolve_function(&self, name: &Ident) -> Result<Box<Type>, TypeError> {
        match self.env.get(name) {
            Some(ty) => match &ty.binding.ty {
                BindingType::Function(_fn_attr) => match ty.ty.clone() {
                    Type::Function(f) => Ok(P(Type::Function(f))),
                    _ => Err(TypeError {
                        message: format!("Expected function, got {:?}", ty),
                    }),
                },
                e => Err(TypeError {
                    message: format!("Expected function, got {:?}", e),
                }),
            },
            None => Err(TypeError {
                message: format!("Undefined function {}", name),
            }),
        }
    }

    pub fn infer_item_type(&mut self, item: &Item) -> Result<Type, TypeError> {
        match &item.kind {
            ItemKind::Fn {
                name,
                attrs,
                function,
                body,
            } => todo!(),
            ItemKind::Bind {
                name,
                attrs,
                type_annotation,
                initializer,
            } => todo!(),
            ItemKind::Effect {
                name,
                generic_params,
                bounds,
                where_clause,
                members,
            } => todo!(),
            ItemKind::Struct {
                name,
                generic_params,
                where_clause,
                members,
            } => todo!(),
            ItemKind::Enum {
                name,
                generic_params,
                where_clause,
                variants,
            } => todo!(),
            ItemKind::Trait {
                name,
                generic_params,
                bounds,
                where_clause,
                members,
            } => todo!(),
            ItemKind::Union {
                name,
                generic_params,
                bounds,
                where_clause,
            } => todo!(),
            ItemKind::Impl {
                name,
                generic_params,
                kind,
                target,
                target_generic_params,
                bounds,
                where_clause,
                members,
            } => todo!(),
        }
    }

    pub fn infer_expr_type(&mut self, expr: &Expr) -> Result<Type, TypeError> {
        match &expr.kind {
            ExprKind::Array(thin_vec) => todo!(),
            ExprKind::ConstBlock(_) => todo!(),
            ExprKind::Call {
                callee,
                generic_args,
                args,
            } => todo!(),
            ExprKind::MethodCall {
                path_seg,
                receiver,
                args,
            } => todo!(),
            ExprKind::Binary { binop, lhs, rhs } => todo!(),
            ExprKind::Unary(unary_operator, expr) => todo!(),
            ExprKind::Cast(expr, ty) => todo!(),
            ExprKind::Literal(literal) => todo!(),
            ExprKind::Let { pat, ty, init } => todo!(),
            ExprKind::Type { expr, ty } => todo!(),
            ExprKind::If { cond, then, else_ } => todo!(),
            ExprKind::While { cond, body, label } => todo!(),
            ExprKind::For {
                pat,
                iter,
                body,
                label,
            } => todo!(),
            ExprKind::Loop { body, label } => todo!(),
            ExprKind::Match { expr, arms } => todo!(),
            ExprKind::Block(block, _) => todo!(),
            ExprKind::Await(expr) => todo!(),
            ExprKind::Guard { condition, body } => todo!(),
            ExprKind::Assign { lhs, rhs } => todo!(),
            ExprKind::AssignOp { lhs, op, rhs } => todo!(),
            ExprKind::Closure {
                callee,
                params,
                closure,
            } => todo!(),
            ExprKind::TrailingClosure {
                callee,
                args,
                closure,
            } => todo!(),
            ExprKind::Struct {
                qual_self,
                path,
                fields,
            } => todo!(),
            ExprKind::PipelineOperation { prev, next } => todo!(),
            ExprKind::Field(expr, _) => todo!(),
            ExprKind::Index { expr, index } => todo!(),
            ExprKind::Range { start, end, limits } => todo!(),
            ExprKind::Underscore => todo!(),
            ExprKind::Paren(expr) => todo!(),
            ExprKind::Path(qualified_self, path) => todo!(),
            ExprKind::Break { label, expr } => todo!(),
            ExprKind::Continue { label } => todo!(),
            ExprKind::Return(expr) => todo!(),
            ExprKind::Try(expr) => todo!(),
            ExprKind::Unwrap(expr) => todo!(),
            ExprKind::Run(expr) => todo!(),
        }
    }

    pub fn typecheck_statement(&mut self, statement: &Statement) -> Result<Type, TypeError> {
        match &statement.kind {
            StatementKind::Let(local) => todo!(),
            StatementKind::Item(item) => todo!(),
            StatementKind::Expr(expr) => todo!(),
            StatementKind::Semicolon(expr) => todo!(),
            StatementKind::Empty => todo!(),
        }
    }

    #[instrument]
    pub fn infer_type(&mut self, ast: &AstElem) -> Result<Type, TypeError> {
        match &ast.kind {
            AstElemKind::Program(thin_vec) => {
                let mut last_type = Type::Infer;
                for statement in thin_vec.iter() {
                    last_type = self.infer_type(statement)?;
                }
                Ok(last_type)
            }
            AstElemKind::Expr(expr) => self.infer_expr_type(expr),
            AstElemKind::Item(item) => self.infer_item_type(item),
            AstElemKind::Statement(statement) => self.typecheck_statement(statement),
        }

        // match ast {
        //     AstNode::Program(program) => {
        //         let mut last_type = Type::Infer;
        //         for statement in program.iter() {
        //             last_type = self.infer_type(statement)?;
        //         }
        //         Ok(last_type)
        //     }
        //     AstNode::FunctionDeclaration {
        //         name,
        //         attrs,
        //         function:
        //             crate::ast::ty::Function {
        //                 generic_params,
        //                 inputs,
        //                 output,
        //             },
        //         body,
        //     } => {
        //         let mut fn_checker = self.copy_env();
        //         for generic_param in generic_params.iter() {
        //             match &generic_param.kind {
        //                 crate::ast::ty::GenericParamKind::Type(Some(box ty)) => {
        //                     let ty = P(Type::from(ty.clone()));
        //                     fn_checker.insert_type(&generic_param.name, &ty);
        //                 }
        //                 crate::ast::ty::GenericParamKind::Type(None) => {
        //                     fn_checker.insert_type(&generic_param.name, &Type::Infer);
        //                 }
        //                 crate::ast::ty::GenericParamKind::Const { box ty, .. } => {
        //                     let ty = P(Type::from(ty.clone()));
        //                     fn_checker.insert_type(&generic_param.name, &ty);
        //                 }
        //             }
        //         }
        //         let param_types: ThinVec<_> = inputs
        //             .iter()
        //             .map(|param| {
        //                 let ty = P(Type::from(*param.ty.clone()));
        //                 fn_checker.insert_type(&param.name, &ty);
        //                 ty
        //             })
        //             .collect();
        //         let mut return_type = Type::from(output.clone());
        //         let attrs = FnAttr::from_list(attrs);
        //         let fn_type = Type::Function(Function {
        //             generic_params: generic_params.clone().into_iter().map(Into::into).collect(),
        //             inputs: inputs
        //                 .iter()
        //                 .zip(param_types.iter())
        //                 .map(|(t1, t2)| Param {
        //                     name: t1.name.clone(),
        //                     ty: P(*t2.clone()),
        //                 })
        //                 .collect(),
        //             // TODO: Placeholder for now
        //             attrs: attrs.clone(),
        //             output: P(return_type.clone()),
        //         });

        //         fn_checker.insert_function(name, &fn_type, attrs.clone(), None);
        //         for statement in body.iter() {
        //             if let &box AstNode::ReturnStatement(ref expr) = statement {
        //                 let statement_type = if let Some(expr) = expr {
        //                     fn_checker.infer_type(expr).inspect_err(|e|{error!(%e);})?
        //                 } else {
        //                     Type::unit()
        //                 };
        //                 if statement_type.infer_or_type(&return_type)
        //                     != return_type.infer_or_type(&statement_type)
        //                 {
        //                     return Err(TypeError {
        //                             message: format!("return statement has type: {:?} but function specifies type: {:?}",
        //                                 statement_type, return_type)
        //                         });
        //                 }
        //                 return_type = statement_type.infer_or_type(&return_type).clone();
        //                 break;
        //             }
        //             let statement_type = fn_checker.infer_type(statement)
        //                 .inspect_err(|e|{error!(%e);})?;
        //             fn_checker.add_anon_type(&statement_type);
        //         }
        //         let fn_type = Type::Function(Function {
        //             generic_params: generic_params.clone().into_iter().map(Into::into).collect(),
        //             inputs: inputs
        //                 .iter()
        //                 .zip(param_types.iter())
        //                 .map(|(t1, t2)| Param {
        //                     name: t1.name.clone(),
        //                     ty: P(*t2.clone()),
        //                 })
        //                 .collect(),
        //             // TODO: Placeholder for now
        //             attrs: attrs.clone(),
        //             output: P(return_type),
        //         });
        //         self.insert_function(&name, &fn_type, attrs, None);
        //         Ok(fn_type)
        //     }
        //     AstNode::VariableDeclaration {
        //         name,
        //         attrs,
        //         type_annotation,
        //         initializer,
        //     } => {
        //         let bind_attr = BindAttr::from_list(attrs);
        //         let var_type = annotation_to_type(type_annotation);
        //         if let Some(initializer) = initializer {
        //             if var_type == Type::Infer {
        //                 let inferred_type = self.infer_type(initializer)
        //                     .inspect_err(|e|{error!(%e);})?;
        //                 self.insert_variable(&name, &inferred_type, bind_attr);
        //                 Ok(inferred_type)
        //             } else {
        //                 let inferred_type = self.infer_type(initializer)
        //                     .inspect_err(|e|{error!(%e);})?;
        //                 if inferred_type != Type::Infer && inferred_type != var_type {
        //                     Err(TypeError {
        //                         message: format!(
        //                             "Type mismatch: expected {:?}, got {:?}",
        //                             var_type, inferred_type
        //                         ),
        //                     })
        //                 } else if inferred_type == var_type {
        //                     self.insert_variable(&name, &inferred_type, bind_attr);
        //                     Ok(var_type)
        //                 } else {
        //                     // currently an error, but we can likely recover by inferring the type from other usages
        //                     Err(TypeError {
        //                         message: format!(
        //                             "Couldn't infer type for variable {} from initializer {:?}",
        //                             name, initializer
        //                         ),
        //                     })
        //                 }
        //             }
        //         } else {
        //             self.insert_variable(&name, &var_type, bind_attr);
        //             Ok(var_type)
        //         }
        //     }
        //     AstNode::IfStatement {
        //         condition,
        //         then_branch,
        //         else_branch,
        //     } => {
        //         let cond_type = self.infer_type(condition).inspect_err(|e|{error!(%e);})?;
        //         if cond_type != Type::Bool {
        //             Err(TypeError {
        //                 message: format!(
        //                     "conditional expression `{:?}` should resolve to a bool",
        //                     cond_type
        //                 ),
        //             })
        //         } else {
        //             let then_type = self.infer_type(then_branch).inspect_err(|e|{error!(%e);})?;
        //             if let Some(else_branch) = else_branch {
        //                 let else_type = self.infer_type(else_branch).inspect_err(|e|{error!(%e);})?;
        //                 if then_type != else_type {
        //                     Err(TypeError {
        //                         message: format!("then and else branches of if statement should have the same type,
        //                             but they have different types: {:?} and {:?}", then_type, else_type) })
        //                 } else {
        //                     Ok(then_type)
        //                 }
        //             } else {
        //                 Ok(then_type)
        //             }
        //         }
        //     }
        //     AstNode::WhileLoop { condition, body } => {
        //         let cond_type = self.infer_type(condition).inspect_err(|e|{error!(%e);})?;
        //         if cond_type != Type::Bool {
        //             Err(TypeError {
        //                 message: format!(
        //                     "conditional expression `{:?}` should resolve to a bool",
        //                     cond_type
        //                 ),
        //             })
        //         } else {
        //             self.infer_type(body)
        //         }
        //     }
        //     AstNode::ForInLoop {
        //         item,
        //         iterable,
        //         body,
        //     } => {
        //         let iter_type = self.infer_type(iterable).inspect_err(|e|{error!(%e);})?;
        //         if let Type::Array(ty) = iter_type {
        //             let mut loop_checker = self.copy_env();
        //             loop_checker.insert_variable(&item, &ty,
        //                 BindAttr::new(false, Some(RefKind::Sync(Mutability::Not)))
        //             );
        //             loop_checker.infer_type(body)
        //         } else {
        //             // For now until we have other things to iterate over
        //             Err(TypeError {
        //                 message: format!("iterable `{:?}` should resolve to an array", iter_type),
        //             })
        //         }
        //     }
        //     AstNode::GuardStatement { condition, body } => {
        //         let cond_type = self.infer_type(condition).inspect_err(|e|{error!(%e);})?;
        //         if cond_type != Type::Bool {
        //             Err(TypeError {
        //                 message: format!(
        //                     "conditional expression `{:?}` should resolve to a bool",
        //                     cond_type
        //                 ),
        //             })
        //         } else {
        //             let body_type = self.infer_type(body).inspect_err(|e|{error!(%e);})?;
        //             Ok(body_type)
        //         }
        //     }
        //     AstNode::ReturnStatement(ast_node) => {
        //         if let Some(expr) = ast_node {
        //             self.infer_type(expr).inspect_err(|e|{error!(%e);})
        //         } else {
        //             Ok(Type::unit())
        //         }
        //     }
        //     AstNode::Block(block) => {
        //         let mut last_type = Type::Infer;
        //         for statement in block.iter() {
        //             last_type = self.infer_type(statement)?;
        //         }
        //         Ok(last_type)
        //     }
        //     AstNode::BinaryOperation {
        //         left,
        //         operator,
        //         right,
        //     } => {
        //         let left_type = self.infer_type(left).inspect_err(|e|{error!(%e);})?;
        //         let right_type = self.infer_type(right).inspect_err(|e|{error!(%e);})?;
        //         self.typecheck_binary_op(operator, &left_type, &right_type)
        //     }
        //     AstNode::UnaryOperation { operator, operand } => {
        //         let operand_type = self.infer_type(operand).inspect_err(|e|{error!(%e);})?;
        //         self.typecheck_unary_op(operator, &operand_type)
        //     }
        //     AstNode::FunctionCall { callee, arguments } => {
        //         let callee_type = self.infer_type(callee).inspect_err(|e|{error!(%e);})?;
        //         if let Type::Function(function) = callee_type {
        //             if arguments.len() != function.inputs.len() {
        //                 return Err(TypeError {
        //                     message: format!(
        //                         "Expected {} arguments, got {}",
        //                         function.inputs.len(),
        //                         arguments.len()
        //                     ),
        //                 });
        //             }
        //             let return_type = function.output.clone();
        //             for (argument, param_type) in arguments.iter().zip(function.inputs.iter()) {
        //                 let arg_type = self.infer_type(argument).inspect_err(|e|{error!(%e);})?;
        //                 if arg_type != *param_type.ty {
        //                     return Err(TypeError {
        //                         message: format!(
        //                             "Argument type mismatch: expected {:?}, got {:?}",
        //                             param_type, arg_type
        //                         ),
        //                     });
        //                 }
        //             }
        //             Ok(*return_type)
        //         } else {
        //             Err(TypeError {
        //                 message: format!("Expected function, got {:?}", callee_type),
        //             })
        //         }
        //     }
        //     AstNode::GenericFunctionCall {
        //         name,
        //         generic_args,
        //         arguments,
        //     } => {
        //         let callee_type = self.resolve_function(name)?;
        //         match *callee_type {
        //             Type::Function(function) => {
        //                 self.typecheck_function_call(&function, arguments, generic_args)
        //                     .inspect_err(|e|{error!(%e);})
        //             }
        //             e => Err(TypeError {
        //                 message: format!("Expected function, got {:?}, error: {:?}", name, e),
        //             }),
        //         }
        //     }
        //     AstNode::TrailingClosure { callee, closure } => {
        //         let mut full_args: ThinVec<Box<AstNode>> = thin_vec![];
        //         let mut generic_arguments: ThinVec<Box<Ty>> = thin_vec![];
        //         let callee_type = match *callee {
        //             box AstNode::GenericFunctionCall {
        //                 ref name,
        //                 ref generic_args,
        //                 ref arguments,
        //             } => {
        //                 full_args.append(&mut arguments.clone());
        //                 // Currently assuming the last argument is the closure, as in Kotlin
        //                 full_args.push(closure.clone());
        //                 generic_arguments.append(&mut generic_args.clone());
        //                 self.resolve_function(name).inspect_err(|e|{error!(%e);})?
        //             }
        //             box AstNode::FunctionCall {
        //                 ref callee,
        //                 ref arguments,
        //             } => {
        //                 full_args.append(&mut arguments.clone());
        //                 // Currently assuming the last argument is the closure, as in Kotlin
        //                 full_args.push(closure.clone());
        //                 P(self.infer_type(callee).inspect_err(|e|{error!(%e);})?)
        //             }
        //             box AstNode::Identifier(ref name) => {
        //                 P(self.resolve_ident(name)?.ty)
        //             }
        //             box ref e => {
        //                 return Err(TypeError {
        //                     message: format!(
        //                         "Expected function call, got {:?}, error: {:?}",
        //                         callee, e
        //                     ),
        //                 })
        //             }
        //         };
        //         match *callee_type {
        //             Type::Function(function) => {
        //                 self.typecheck_function_call(&function, &full_args, &generic_arguments)
        //             }
        //             e => Err(TypeError {
        //                 message: format!("Expected function, got {:?}, error: {:?}", callee, e),
        //             }),
        //         }
        //     }
        //     AstNode::PipelineOperation { prev, next } => {
        //         //TODO: Placeholder implementation
        //         let prev_type = self.infer_type(prev)?;
        //         let next_type = self.infer_type(next)?;
        //         //self.typecheck_binary_op(&BinaryOperator::Pipeline, &left_type, &right_type)
        //         Ok(prev_type)
        //     }
        //     AstNode::Identifier(ident) => {
        //         Ok(self.resolve_ident(ident).inspect_err(|e|{error!(%e);})?.ty)
        //     }
        //     AstNode::IntLiteral(_) => Ok(Type::Int(IntKind::Int)),
        //     AstNode::FloatLiteral(_) => Ok(Type::Float(FloatKind::Float)),
        //     AstNode::StringLiteral(_) => Ok(Type::String),
        //     AstNode::BoolLiteral(_) => Ok(Type::Bool),
        //     AstNode::ArrayLiteral(elements) => self.typecheck_array_literal(elements),
        //     AstNode::EffectDeclaration {
        //         name, generic_params,
        //         where_clause, bounds, members } => todo!(),
        //     AstNode::StructDeclaration {
        //         name, generic_params,
        //         where_clause, members
        //     } => todo!(),
        //     AstNode::EnumDeclaration {
        //         name, generic_params,
        //         where_clause, variants
        //     } => todo!(),
        //     AstNode::TraitDeclaration {
        //         name, generic_params,
        //         bounds, where_clause, members
        //     } => todo!(),
        //     AstNode::UnionDeclaration {
        //         name, generic_params,
        //         bounds, where_clause
        //     } => todo!(),
        //     AstNode::ImplDeclaration {
        //         name, generic_params,
        //         kind,
        //         target, target_generic_params,
        //         where_clause,
        //         bounds, members
        //     } => todo!(),
        //     AstNode::WithClause(items) => todo!(),
        // }
    }

    #[instrument]
    fn typecheck_function_call(
        &mut self,
        function: &Function,
        arguments: &ThinVec<Box<AstElem>>,
        generic_args: &ThinVec<Box<Ty>>,
    ) -> Result<Type, TypeError> {
        if arguments.len() != function.inputs.len() {
            return Err(TypeError {
                message: format!(
                    "Expected {} arguments, got {}",
                    function.inputs.len(),
                    arguments.len()
                ),
            });
        }
        let generic_args: ThinVec<_> = generic_args
            .iter()
            .map(|ty| P(Type::from(*ty.clone())))
            .collect();
        let mut generic_checker = self.copy_env();
        if generic_args.len() != function.generic_params.len() {
            // TODO: figure out if we should be able to infer generic arguments
            return Err(TypeError {
                message: format!(
                    "Expected {} generic arguments, got {}",
                    function.generic_params.len(),
                    generic_args.len()
                ),
            });
        }
        let generic_params = function.generic_params.clone();
        for (generic_arg, generic_param) in generic_args.iter().zip(generic_params.iter()) {
            match &generic_param.kind {
                GenericParamKind::Type(Some(box ty)) => {
                    if **generic_arg != *ty {
                        return Err(TypeError {
                            message: format!(
                                "Generic argument type mismatch: expected {:?}, got {:?}",
                                ty, generic_arg
                            ),
                        });
                    }
                    generic_checker.insert_type(&generic_param.name, ty);
                }
                GenericParamKind::Type(None) => {
                    generic_checker.insert_type(&generic_param.name, &Type::Infer);
                }
                GenericParamKind::Const {
                    box ty,
                    value: None,
                } => {
                    if **generic_arg != *ty {
                        return Err(TypeError {
                            message: format!(
                                "Generic argument type mismatch: expected {:?}, got {:?}",
                                ty, generic_arg
                            ),
                        });
                    }
                    generic_checker.insert_type(&generic_param.name, ty);
                }
                GenericParamKind::Const {
                    box ty,
                    value: Some(value),
                } => {
                    if **generic_arg != *ty {
                        return Err(TypeError {
                            message: format!(
                                "Generic argument type mismatch: expected {:?}, got {:?}",
                                ty, generic_arg
                            ),
                        });
                    }
                    generic_checker
                        .insert_type(&generic_param.name, &P(Type::Const(value.clone())));
                }
            }
        }
        let return_type = function.output.clone();
        for (argument, param_type) in arguments.iter().zip(function.inputs.iter()) {
            let arg_type = generic_checker.infer_type(argument)?;
            if arg_type != *param_type.ty {
                return Err(TypeError {
                    message: format!(
                        "Argument type mismatch: expected {:?}, got {:?}",
                        param_type, arg_type
                    ),
                });
            }
        }
        Ok(*return_type)
    }
    /// Checks the types for a binary operation.
    #[instrument]
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
                if right == &Type::Infer
                    && (matches!(left, &Type::Int(_)) || matches!(left, &Type::Float(_)))
                {
                    Ok(right.clone())
                } else if left == &Type::Infer
                    && (matches!(right, &Type::Int(_)) || matches!(right, &Type::Float(_)))
                    || left == right
                        && (matches!(left, &Type::Int(_)) || matches!(left, &Type::Float(_)))
                {
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
                if right == &Type::Infer
                    && (matches!(left, &Type::Int(_)) || matches!(left, &Type::Float(_)))
                    || left == &Type::Infer
                        && (matches!(right, &Type::Int(_)) || matches!(right, &Type::Float(_)))
                    || left == right
                        && (matches!(left, &Type::Int(_)) || matches!(left, &Type::Float(_)))
                {
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
            }
            BinaryOperator::Pipeline => todo!(),
        }
    }

    /// Checks the types for a unary operation.
    #[instrument]
    fn typecheck_unary_op(&self, op: &UnaryOperator, operand: &Type) -> Result<Type, TypeError> {
        match op {
            UnaryOperator::Negate => {
                if matches!(operand, &Type::Int(_)) || matches!(operand, &Type::Float(_)) {
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

    #[instrument]
    pub fn typecheck_array_literal(
        &mut self,
        elements: &ThinVec<Box<AstElem>>,
    ) -> Result<Type, TypeError> {
        if elements.is_empty() {
            return Ok(Type::Array(P(Type::Infer)));
        }
        let mut first_type = self
            .infer_type(elements.first().unwrap())
            .inspect_err(|e| {
                error!(%e);
            })?;
        for element in elements.iter().skip(1) {
            let element_type = self.infer_type(element).inspect_err(|e| {
                error!(%e);
            })?;
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

    #[instrument(skip(self))]
    pub fn typecheck_program(&mut self, program: &AstElem) -> Result<(), TypeError> {
        self.infer_type(program)?;
        Ok(())
    }
}

/// Type checks an AST node.
pub fn typecheck(ast: &AstElem) -> Result<(), TypeError> {
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

#[derive(Debug, Clone, PartialEq)]
pub struct TypeBinding {
    pub ty: Type,
    pub binding: Binding,
}

impl TypeBinding {
    pub fn new(ty: Type, binding: Binding) -> Self {
        TypeBinding { ty, binding }
    }

    pub fn variable(ty: Type, binding: BindAttr) -> Self {
        TypeBinding::new(
            ty,
            Binding {
                visibility: Visibility::Local(None),
                ty: BindingType::Value(binding),
            },
        )
    }

    pub fn function(ty: Type, binding: FnAttr, visibility: Option<Visibility>) -> Self {
        TypeBinding::new(
            ty,
            Binding {
                visibility: visibility.unwrap_or(Visibility::Private),
                ty: BindingType::Function(binding),
            },
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Binding {
    pub visibility: Visibility,
    pub ty: BindingType,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BindingType {
    Value(BindAttr),
    Ref(RefKind, BindAttr),
    Function(FnAttr),
}

/// Represents a type in the Alloy type system.
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Int(IntKind),
    UInt(UintKind),
    Float(FloatKind),
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
            _ => self,
        }
    }

    pub fn normalize(self) -> Self {
        match self {
            Type::Simple(name) => match name.as_str() {
                "int" => Type::Int(IntKind::Int),
                "float" => Type::Float(FloatKind::Float),
                "bool" => Type::Bool,
                "char" => Type::Char,
                "string" => Type::String,
                "Array" => Type::Array(P(Type::Infer)),
                _ => Type::Simple(name),
            },
            Type::Generic(name, types) => match name.as_str() {
                "Array" => {
                    let elem_type = types
                        .first()
                        .map(|ty| ty.clone().normalize())
                        .unwrap_or(Type::Infer);
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
            TyKind::Uint(ty) => Type::UInt(ty),
            TyKind::Float(ty) => Type::Float(ty),
            TyKind::Bool => Type::Bool,
            TyKind::Char => Type::Char,
            TyKind::String => Type::String,
            TyKind::Array(ty) => Type::Array(P(Type::from(*ty))),
            TyKind::Path(path) => Type::Path(path),
            TyKind::Tuple(types) => Type::Tuple(
                types
                    .into_iter()
                    .map(|ty| P(Type::from(*ty).normalize()))
                    .collect(),
            ),
            TyKind::SizedArray(ty, size) => Type::SizedArray(P(Type::from(*ty)), size),
            TyKind::Function(function) => Type::Function(function.into()),
            TyKind::Const(const_) => Type::Const(const_),
            TyKind::Algebraic(type_op) => Type::Algebraic(type_op.into()),
            TyKind::Generic(name, types) => Type::Generic(
                name,
                types
                    .into_iter()
                    .map(|ty| P(Type::from(*ty).normalize()))
                    .collect(),
            )
            .normalize(),
            TyKind::Paren(ty) => Type::Paren(P(Type::from(*ty).normalize())),
            TyKind::Ref(kind, ty) => Type::Ref(kind, P(Type::from(*ty).normalize())),
            TyKind::Pattern(ty, pat) => {
                Type::Pattern(P(Type::from(*ty).normalize()), P(PatternType::from(*pat)))
            }
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
                types
                    .into_iter()
                    .map(|ty| P(Type::from(*ty).normalize()))
                    .collect(),
            ),
            TypeOp::Or(types) => AlgebraicType::Or(
                types
                    .into_iter()
                    .map(|ty| P(Type::from(*ty).normalize()))
                    .collect(),
            ),
            TypeOp::Xor(types) => AlgebraicType::Xor(
                types
                    .into_iter()
                    .map(|ty| P(Type::from(*ty).normalize()))
                    .collect(),
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
    Ident(BindAttr, Ident, Option<Box<PatternType>>),
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
            PatternKind::Ident(mode, name, pat) => {
                PatternType::Ident(mode, name, pat.map(|pat| P(PatternType::from(*pat))))
            }
            PatternKind::Tuple(patterns) => PatternType::Tuple(
                patterns
                    .into_iter()
                    .map(|pat| P(PatternType::from(*pat)))
                    .collect(),
            ),
            PatternKind::TupleStruct(qual_self, path, patterns) => PatternType::TupleStruct(
                qual_self.map(|qual_self| P(QualifiedSelf::from(*qual_self))),
                path,
                patterns
                    .into_iter()
                    .map(|pat| P(PatternType::from(*pat)))
                    .collect(),
            ),
            PatternKind::Struct(qual_self, path, patterns) => PatternType::Struct(
                qual_self.map(|qual_self| P(QualifiedSelf::from(*qual_self))),
                path,
                patterns.into_iter().map(PatternField::from).collect(),
            ),
            PatternKind::Path(qual_self, path) => PatternType::Path(
                qual_self.map(|qual_self| P(QualifiedSelf::from(*qual_self))),
                path,
            ),
            PatternKind::Ref(pat, kind) => PatternType::Ref(P(PatternType::from(*pat)), kind),
            PatternKind::Or(patterns) => PatternType::Or(
                patterns
                    .into_iter()
                    .map(|pat| P(PatternType::from(*pat)))
                    .collect(),
            ),
            PatternKind::TypeOp(type_op, pat) => {
                PatternType::Algebraic(type_op.into(), P(PatternType::from(*pat)))
            }
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

impl From<crate::ast::ty::QualifiedSelf> for QualifiedSelf {
    fn from(qual_self: crate::ast::ty::QualifiedSelf) -> Self {
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
    pub attrs: FnAttr,
    pub inputs: ThinVec<Param>,
    pub output: Box<Type>,
}

impl From<crate::ast::ty::Function> for Function {
    fn from(function: crate::ast::ty::Function) -> Self {
        Self {
            generic_params: function
                .generic_params
                .into_iter()
                .map(Into::into)
                .collect(),
            inputs: function.inputs.into_iter().map(Into::into).collect(),
            attrs: FnAttr {
                is_async: false,
                is_shared: false,
                effects: ThinVec::new(),
            },
            output: P(Type::from(function.output)),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FnAttr {
    pub is_async: bool,
    pub is_shared: bool,
    pub effects: ThinVec<Box<WithClauseItem>>,
}

impl FnAttr {
    pub fn from_list(attrs: &[crate::ast::FnAttr]) -> Self {
        let mut is_async = false;
        let mut is_shared = false;
        let mut effects = ThinVec::new();
        for attr in attrs {
            is_async = is_async || attr.is_async;
            is_shared = is_shared || attr.is_shared;
            let mut e: ThinVec<_> = attr
                .effects
                .iter()
                .map(|e| P(WithClauseItem::from(*e.clone())))
                .collect();
            effects.append(&mut e);
        }
        FnAttr {
            is_async,
            is_shared,
            effects,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum WithClauseItem {
    Generic(GenericParam),
    Algebraic(AlgebraicType),
}

impl From<crate::ast::WithClauseItem> for WithClauseItem {
    fn from(item: crate::ast::WithClauseItem) -> Self {
        match item {
            crate::ast::WithClauseItem::Generic(generic_param) => {
                WithClauseItem::Generic(GenericParam::from(generic_param))
            }
            crate::ast::WithClauseItem::Algebraic(op) => {
                WithClauseItem::Algebraic(AlgebraicType::from(op))
            }
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

impl From<crate::ast::ty::GenericParam> for GenericParam {
    fn from(param: crate::ast::ty::GenericParam) -> Self {
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
    Const { ty: Box<Type>, value: Option<Const> },
}

impl From<crate::ast::ty::GenericParamKind> for GenericParamKind {
    fn from(kind: crate::ast::ty::GenericParamKind) -> Self {
        match kind {
            crate::ast::ty::GenericParamKind::Type(ty) => {
                Self::Type(ty.map(|ty| P(Type::from(*ty).normalize())))
            }
            crate::ast::ty::GenericParamKind::Const { ty, value } => Self::Const {
                ty: P(Type::from(*ty).normalize()),
                value,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: Ident,
    pub ty: Box<Type>,
}

impl From<crate::ast::ty::Param> for Param {
    fn from(param: crate::ast::ty::Param) -> Self {
        Param {
            name: param.name,
            ty: P(Type::from(*param.ty).normalize()),
        }
    }
}
