//! Transpiler for the Alloy programming language.
//!
//! This module is responsible for converting the Alloy Abstract Syntax Tree (AST)
//! into equivalent Rust code. It traverses the AST and generates Rust syntax
//! for each Alloy language construct, ensuring that the semantics of the original
//! Alloy code are preserved in the generated Rust code.

use thin_vec::ThinVec;

use crate::{
    ast::{
        ty::{Const, FnRetTy, IntKind, Mutability, Pattern, PatternKind, Ty, UintKind},
        AstElem, AstElemKind, BinaryOperator, Expr, ExprKind, Item, ItemKind, Literal, LocalKind,
        Statement, StatementKind, UnaryOperator, P,
    },
    error::TranspilerError,
    type_checker::{Param, Type},
};

/// Represents the transpiler for converting Alloy AST to Rust code.
#[derive(Default)]
pub struct Transpiler {
    indent_level: usize,
}

impl Transpiler {
    /// Creates a new Transpiler instance.
    pub fn new() -> Self {
        Transpiler { indent_level: 0 }
    }

    /// Generates the current indentation string based on the indent level.
    fn indent(&self) -> String {
        "    ".repeat(self.indent_level)
    }

    /// Main entry point for transpiling an AST node to Rust code.
    ///
    /// This method dispatches to specific transpilation methods based on the type of the AST node.
    pub fn transpile(&mut self, node: &AstElem) -> String {
        match &node.kind {
            AstElemKind::Program(statements) => self.transpile_program(&statements[..]),
            AstElemKind::Expr(expr) => self.transpile_expr(expr),
            AstElemKind::Item(item) => self.transpile_item(item),
            AstElemKind::Statement(statement) => self.transpile_statement(statement),
        }
    }

    fn transpile_item(&mut self, item: &Item) -> String {
        match &item.kind {
            ItemKind::Bind {
                name,
                attrs,
                type_annotation,
                initializer,
            } => self.transpile_variable_declaration(
                name.as_str(),
                attrs
                    .first()
                    .map_or(false, |attr| attr.mutability == Mutability::Mut),
                &type_annotation.as_ref().map(|ty| Type::from(*ty.clone())),
                &initializer,
            ),
            ItemKind::Fn {
                name,
                attrs,
                function,
                body,
            } => self.transpile_function(
                &name,
                &function
                    .inputs
                    .iter()
                    .map(|param| param.clone().into())
                    .collect::<Vec<_>>(),
                &function.output.to_type(),
                &AstElem::expr(P(Expr::block(body.clone(), None))),
            ),
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
                where_clause,
                bounds,
                members,
            } => todo!(),
        }
    }
    fn transpile_expr(&mut self, expr: &Expr) -> String {
        match &expr.kind {
            ExprKind::Array(elements) => self.transpile_array_literal(&elements),
            ExprKind::ConstBlock(_) => todo!(),
            ExprKind::Call {
                callee,
                generic_args,
                args,
            } => self.transpile_function_call(
                callee,
                &args.iter().map(|arg| &**arg).collect::<Vec<_>>()[..],
            ),
            ExprKind::MethodCall {
                path_seg,
                receiver,
                args,
            } => todo!(),
            ExprKind::Binary { binop, lhs, rhs } => self.transpile_binary_op(binop, lhs, rhs),
            ExprKind::Unary(unary_operator, operand) => {
                self.transpile_unary_op(unary_operator, operand)
            }
            ExprKind::Cast(expr, ty) => todo!(),
            ExprKind::Literal(literal) => literal.to_string(),
            ExprKind::Let { pat, ty, init } => {
                let pat_str = self.transpile_pattern(pat);
                let ty_str = ty
                    .as_ref()
                    .map_or(String::new(), |ty| format!(": {}", self.transpile_expr(ty)));
                let init_str = init
                    .as_ref()
                    .map_or(String::new(), |init| self.transpile_expr(init));
                format!("{}{}{};", pat_str, ty_str, init_str)
            }
            ExprKind::Type { expr, ty } => todo!(),
            ExprKind::Guard { condition, body } => todo!(),
            ExprKind::If { cond, then, else_ } => self.transpile_if(cond, then, else_),
            ExprKind::While { cond, body, label } => self.transpile_while(cond, body),
            ExprKind::For {
                pat,
                iter,
                body,
                label,
            } => self.transpile_for(&Some(P(*pat.clone())), &Some(P(*iter.clone())), &None, body),
            ExprKind::Loop { body, label } => todo!(),
            ExprKind::Match { expr, arms } => todo!(),
            ExprKind::Block(block, _) => self.transpile_block(&block.stmts[..]),
            ExprKind::Await(expr) => todo!(),
            ExprKind::Assign { lhs, rhs } => {
                let lhs_str = self.transpile_expr(lhs);
                let rhs_str = self.transpile_expr(rhs);
                format!("{} = {};", lhs_str, rhs_str)
            }
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
            ExprKind::Path(qualified_self, path) => {
                let path_str = path.segments.join("::");
                match qualified_self {
                    Some(qual_self) => format!("{}::{}", qual_self, path_str),
                    None => path_str,
                }
            }
            ExprKind::Break { label, expr } => todo!(),
            ExprKind::Continue { label } => todo!(),
            ExprKind::Return(expr) => self.transpile_return(expr),
            ExprKind::Try(expr) => todo!(),
            ExprKind::Unwrap(expr) => todo!(),
            ExprKind::Run(expr) => todo!(),
        }
    }

    fn transpile_statement(&mut self, statement: &Statement) -> String {
        match &statement.kind {
            StatementKind::Let(local) => local.to_string(),
            StatementKind::Item(item) => self.transpile_item(item),
            StatementKind::Expr(expr) => self.transpile_expr(expr),
            StatementKind::Semicolon(expr) => self.transpile_expr(expr),
            StatementKind::Empty => String::new(),
        }
    }
    /// Transpiles the entire program (a list of statements).
    fn transpile_program(&mut self, statements: &[Box<AstElem>]) -> String {
        statements
            .iter()
            .map(|stmt| self.transpile(stmt))
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Transpiles a function declaration.
    fn transpile_function(
        &mut self,
        name: &str,
        params: &[Param],
        return_type: &Option<Type>,
        body: &AstElem,
    ) -> String {
        let params_str = params
            .iter()
            .map(|p| format!("{}: {}", p.name, self.transpile_type(&p.ty)))
            .collect::<Vec<_>>()
            .join(", ");

        let return_type_str = match return_type {
            Some(ty) => format!(" -> {}", self.transpile_type(ty)),
            None => String::new(),
        };

        let body_str = self.transpile(body);

        format!(
            "fn {}({}){} {}",
            name, params_str, return_type_str, body_str
        )
    }

    /// Transpiles a variable declaration.
    fn transpile_variable_declaration(
        &mut self,
        name: &str,
        is_mutable: bool,
        type_annotation: &Option<Type>,
        initializer: &Option<Box<Expr>>,
    ) -> String {
        let mut_keyword = if is_mutable { "mut " } else { "" };
        let type_str = type_annotation
            .as_ref()
            .map(|ty| format!(": {}", self.transpile_type(ty)))
            .unwrap_or_default();

        let init_str = initializer
            .as_ref()
            .map(|expr| format!(" = {}", self.transpile_expr(expr)))
            .unwrap_or_default();

        format!(
            "{}let {}{}{}{};",
            self.indent(),
            mut_keyword,
            name,
            type_str,
            init_str
        )
    }

    fn transpile_pattern(&mut self, pattern: &Pattern) -> String {
        pattern.to_string()
    }

    /// Transpiles a block of statements.
    fn transpile_block(&mut self, statements: &[Statement]) -> String {
        self.indent_level += 1;
        let body = statements
            .iter()
            .map(|stmt| format!("{}{}", self.indent(), self.transpile_statement(stmt)))
            .collect::<Vec<_>>()
            .join("\n");
        self.indent_level -= 1;
        format!("{{\n{}\n{}}}", body, self.indent())
    }

    /// Transpiles a return statement.
    fn transpile_return(&mut self, expr: &Option<Box<Expr>>) -> String {
        match expr {
            Some(e) => format!("{}return {};", self.indent(), self.transpile_expr(e)),
            None => format!("{}return;", self.indent()),
        }
    }

    /// Transpiles an if statement.
    fn transpile_if(
        &mut self,
        condition: &Expr,
        then_branch: &Expr,
        else_branch: &Option<Box<Expr>>,
    ) -> String {
        let else_str = else_branch
            .as_ref()
            .map(|eb| format!(" else {}", self.transpile_expr(eb)))
            .unwrap_or_default();

        format!(
            "{}if {} {} {}",
            self.indent(),
            self.transpile_expr(condition),
            self.transpile_expr(then_branch),
            else_str
        )
    }

    /// Transpiles a while loop.
    fn transpile_while(&mut self, condition: &Expr, body: &Expr) -> String {
        format!(
            "{}while {} {}",
            self.indent(),
            self.transpile_expr(condition),
            self.transpile_expr(body)
        )
    }

    /// Transpiles a for loop.
    fn transpile_for(
        &mut self,
        initializer: &Option<Box<Pattern>>,
        condition: &Option<Box<Expr>>,
        increment: &Option<Box<Expr>>,
        body: &Expr,
    ) -> String {
        let init_str = initializer
            .as_ref()
            .map_or(String::new(), |init| self.transpile_pattern(init));
        let cond_str = condition
            .as_ref()
            .map_or(String::new(), |cond| self.transpile_expr(cond));
        let incr_str = increment
            .as_ref()
            .map_or(String::new(), |incr| self.transpile_expr(incr));

        format!(
            "{}for {}; {}; {} {}",
            self.indent(),
            init_str,
            cond_str,
            incr_str,
            self.transpile_expr(body)
        )
    }

    /// Transpiles a binary operation.
    fn transpile_binary_op(
        &mut self,
        operator: &BinaryOperator,
        left: &Expr,
        right: &Expr,
    ) -> String {
        let left_str = &self.transpile_expr(left);
        let right_str = &self.transpile_expr(right);
        let op_str = match operator {
            BinaryOperator::Add => "+",
            BinaryOperator::Subtract => "-",
            BinaryOperator::Multiply => "*",
            BinaryOperator::Divide => "/",
            BinaryOperator::Equal => "==",
            BinaryOperator::NotEqual => "!=",
            BinaryOperator::LessThan => "<",
            BinaryOperator::GreaterThan => ">",
            BinaryOperator::LessThanOrEqual => "<=",
            BinaryOperator::GreaterThanOrEqual => ">=",
            BinaryOperator::And => "&&",
            BinaryOperator::Or => "||",
            BinaryOperator::Assign => "=",
            BinaryOperator::Pipeline => "|>",
            BinaryOperator::Modulo => "%",
        };
        format!("({} {} {})", left_str, op_str, right_str)
    }

    /// Transpiles a unary operation.
    fn transpile_unary_op(&mut self, operator: &UnaryOperator, operand: &Expr) -> String {
        let op_str = match operator {
            UnaryOperator::Negate => "-",
            UnaryOperator::Not => "!",
            UnaryOperator::Increment => todo!(),
        };

        format!("{}({})", op_str, self.transpile_expr(operand))
    }

    /// Transpiles a function call.
    fn transpile_function_call(&mut self, function: &Expr, arguments: &[&Expr]) -> String {
        let args_str = arguments
            .iter()
            .map(|arg| self.transpile_expr(arg))
            .collect::<Vec<_>>()
            .join(", ");

        format!("{}({})", self.transpile_expr(function), args_str)
    }

    /// Transpiles an array literal.
    fn transpile_array_literal(&mut self, elements: &ThinVec<Box<Expr>>) -> String {
        let elements_str = elements
            .iter()
            .map(|elem| self.transpile_expr(elem))
            .collect::<Vec<_>>()
            .join(", ");

        format!("vec![{}]", elements_str)
    }

    /// Transpiles a type annotation to its Rust equivalent.
    fn transpile_type(&self, type_annotation: &Type) -> String {
        match type_annotation {
            Type::Int(i) => format!("{}", i),
            Type::UInt(u) => format!("{}", u),
            Type::Float(f) => format!("{}", f),
            Type::String => "String".to_string(),
            Type::Bool => "bool".to_string(),
            Type::Simple(name) => name.clone(),
            Type::Array(inner_type) => {
                format!("Vec<{}>", self.transpile_type(inner_type))
            }
            Type::Tuple(thin_vec) => todo!(),
            Type::Function(function) => todo!(),
            Type::Algebraic(type_op) => todo!(),
            Type::Ref(ref_kind, type_annotation) => todo!(),
            Type::Pattern(type_annotation, pattern) => todo!(),
            Type::Char => "char".to_string(),
            Type::Path(path) => todo!(),
            Type::SizedArray(inner_type, size) => {
                let size = match size {
                    Const(box AstElem { id, kind, .. }) => {
                        if let AstElemKind::Expr(expr) = kind {
                            if let ExprKind::Literal(lit) = &expr.kind {
                                if let Literal::Int(value) = lit {
                                    value
                                } else {
                                    todo!()
                                }
                            } else {
                                todo!()
                            }
                        } else {
                            todo!()
                        }
                    }
                    _ => todo!(),
                };
                format!("[{}; {}]", self.transpile_type(inner_type), size)
            }
            Type::Const(_) => todo!(),
            Type::Generic(_, thin_vec) => todo!(),
            Type::Paren(_) => todo!(),
            Type::Any => "Box<dyn Any>".to_string(),
            Type::Infer => "_".to_string(),
            Type::SelfType => todo!(),
            Type::Never => "!".to_string(),
            Type::Err => todo!(),
        }
    }
}

/// Public function to transpile an AST to Rust code.
pub fn transpile(ast: &AstElem) -> Result<String, TranspilerError> {
    let mut transpiler = Transpiler::new();
    Ok(transpiler.transpile(ast))
}
