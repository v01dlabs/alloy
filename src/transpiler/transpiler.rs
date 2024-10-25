//! Transpiler for the Alloy programming language.
//!
//! This module is responsible for converting the Alloy Abstract Syntax Tree (AST)
//! into equivalent Rust code. It traverses the AST and generates Rust syntax
//! for each Alloy language construct, ensuring that the semantics of the original
//! Alloy code are preserved in the generated Rust code.

use thin_vec::ThinVec;

use crate::{
    ast::{ty::{Const, FnRetTy, IntTy, Mutability, Ty}, AstNode, BinaryOperator, UnaryOperator},
    type_checker::{Param, Type},
};

/// Represents the transpiler for converting Alloy AST to Rust code.
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
    pub fn transpile(&mut self, node: &AstNode) -> String {
        match node {
            AstNode::Program(statements) => self.transpile_program(&statements[..]),
            AstNode::FunctionDeclaration {
                name,
                attrs,
                body,
                function,
            } => self.transpile_function(
                name,
                &function
                    .inputs
                    .iter()
                    .map(|param| param.clone().into())
                    .collect::<Vec<_>>(),
                &function.output.to_type(),
                &AstNode::Block(body.clone()),
            ),
            AstNode::VariableDeclaration {
                name,
                    attrs,
                type_annotation,
                initializer,
            } => self.transpile_variable_declaration(
                name,
                attrs.first().map_or(false, |attr| attr.mutability == Mutability::Mut),
                &type_annotation
                    .as_ref()
                    .map(|ty| Type::from(*ty.clone())),
                initializer,
            ),
            AstNode::Block(statements) => self.transpile_block(&statements[..]),
            AstNode::ReturnStatement(expr) => self.transpile_return(expr),
            AstNode::IfStatement {
                condition,
                then_branch,
                else_branch,
            } => self.transpile_if(condition, then_branch, else_branch),
            AstNode::WhileLoop { condition, body } => self.transpile_while(condition, body),
            AstNode::ForInLoop {
                item,
                iterable,
                body,
            } => self.transpile_for(
                &Some(Box::new(AstNode::Identifier(item.clone()))),
                &Some(Box::new(*iterable.clone())),
                &None,
                body,
            ),
            AstNode::BinaryOperation {
                left,
                operator,
                right,
            } => self.transpile_binary_op(left, operator, right),
            AstNode::UnaryOperation { operator, operand } => {
                self.transpile_unary_op(operator, operand)
            }
            AstNode::FunctionCall { callee, arguments } => self.transpile_function_call(
                callee,
                &arguments.iter().map(|arg| &**arg).collect::<Vec<_>>()[..],
            ),
            AstNode::Identifier(name) => name.clone(),
            AstNode::IntLiteral(value) => value.to_string(),
            AstNode::FloatLiteral(value) => value.to_string(),
            AstNode::StringLiteral(value) => format!("\"{}\"", value),
            AstNode::BoolLiteral(value) => value.to_string(),
            AstNode::ArrayLiteral(elements) => self.transpile_array_literal(&elements),
            AstNode::GuardStatement { condition, body } => todo!(),
            AstNode::GenericFunctionCall {
                name,
                generic_args,
                arguments,
            } => todo!(),
            AstNode::TrailingClosure { callee, closure } => todo!(),
            AstNode::PipelineOperation { left, right } => todo!(),
            AstNode::EffectDeclaration { 
                name, generic_params,
                where_clause, bounds, members } => todo!(),
            AstNode::StructDeclaration { 
                name, generic_params, 
                where_clause, members 
            } => todo!(),
            AstNode::EnumDeclaration { 
                name, generic_params, 
                where_clause, variants 
            } => todo!(),
            AstNode::TraitDeclaration { 
                name, generic_params, 
                bounds, where_clause, members 
            } => todo!(),
            AstNode::UnionDeclaration { 
                name, generic_params, 
                bounds, where_clause 
            } => todo!(),
            AstNode::ImplDeclaration { 
                name, generic_params, 
                kind, 
                target, target_generic_params, 
                where_clause,
                bounds, members 
            } => todo!(),
            AstNode::WithClause(items) => todo!(),
        }
    }

    /// Transpiles the entire program (a list of statements).
    fn transpile_program(&mut self, statements: &[Box<AstNode>]) -> String {
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
        body: &AstNode,
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
        initializer: &Option<Box<AstNode>>,
    ) -> String {
        let mut_keyword = if is_mutable { "mut " } else { "" };
        let type_str = type_annotation
            .as_ref()
            .map(|ty| format!(": {}", self.transpile_type(ty)))
            .unwrap_or_default();

        let init_str = initializer
            .as_ref()
            .map(|expr| format!(" = {}", self.transpile(expr)))
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

    /// Transpiles a block of statements.
    fn transpile_block(&mut self, statements: &[Box<AstNode>]) -> String {
        self.indent_level += 1;
        let body = statements
            .iter()
            .map(|stmt| format!("{}{}", self.indent(), self.transpile(stmt)))
            .collect::<Vec<_>>()
            .join("\n");
        self.indent_level -= 1;
        format!("{{\n{}\n{}}}", body, self.indent())
    }

    /// Transpiles a return statement.
    fn transpile_return(&mut self, expr: &Option<Box<AstNode>>) -> String {
        match expr {
            Some(e) => format!("{}return {};", self.indent(), self.transpile(e)),
            None => format!("{}return;", self.indent()),
        }
    }

    /// Transpiles an if statement.
    fn transpile_if(
        &mut self,
        condition: &AstNode,
        then_branch: &AstNode,
        else_branch: &Option<Box<AstNode>>,
    ) -> String {
        let else_str = else_branch
            .as_ref()
            .map(|eb| format!(" else {}", self.transpile(eb)))
            .unwrap_or_default();

        format!(
            "{}if {} {} {}",
            self.indent(),
            self.transpile(condition),
            self.transpile(then_branch),
            else_str
        )
    }

    /// Transpiles a while loop.
    fn transpile_while(&mut self, condition: &AstNode, body: &AstNode) -> String {
        format!(
            "{}while {} {}",
            self.indent(),
            self.transpile(condition),
            self.transpile(body)
        )
    }

    /// Transpiles a for loop.
    fn transpile_for(
        &mut self,
        initializer: &Option<Box<AstNode>>,
        condition: &Option<Box<AstNode>>,
        increment: &Option<Box<AstNode>>,
        body: &AstNode,
    ) -> String {
        let init_str = initializer
            .as_ref()
            .map_or(String::new(), |init| self.transpile(init));
        let cond_str = condition
            .as_ref()
            .map_or(String::new(), |cond| self.transpile(cond));
        let incr_str = increment
            .as_ref()
            .map_or(String::new(), |incr| self.transpile(incr));

        format!(
            "{}for {}; {}; {} {}",
            self.indent(),
            init_str,
            cond_str,
            incr_str,
            self.transpile(body)
        )
    }

    /// Transpiles a binary operation.
    fn transpile_binary_op(
        &mut self,
        left: &AstNode,
        operator: &BinaryOperator,
        right: &AstNode,
    ) -> String {
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
            BinaryOperator::Modulo => todo!(),
            BinaryOperator::Assign => todo!(),
            BinaryOperator::Pipeline => todo!(),
        };

        format!(
            "({} {} {})",
            self.transpile(left),
            op_str,
            self.transpile(right)
        )
    }

    /// Transpiles a unary operation.
    fn transpile_unary_op(&mut self, operator: &UnaryOperator, operand: &AstNode) -> String {
        let op_str = match operator {
            UnaryOperator::Negate => "-",
            UnaryOperator::Not => "!",
            UnaryOperator::Increment => todo!(),
        };

        format!("{}({})", op_str, self.transpile(operand))
    }

    /// Transpiles a function call.
    fn transpile_function_call(&mut self, function: &AstNode, arguments: &[&AstNode]) -> String {
        let args_str = arguments
            .iter()
            .map(|arg| self.transpile(arg))
            .collect::<Vec<_>>()
            .join(", ");

        format!("{}({})", self.transpile(function), args_str)
    }

    /// Transpiles an array literal.
    fn transpile_array_literal(&mut self, elements: &ThinVec<Box<AstNode>>) -> String {
        let elements_str = elements
            .iter()
            .map(|elem| self.transpile(elem))
            .collect::<Vec<_>>()
            .join(", ");

        format!("vec![{}]", elements_str)
    }

    /// Transpiles a type annotation to its Rust equivalent.
    fn transpile_type(&self, type_annotation: &Type) -> String {
        match type_annotation {
            Type::Int(IntTy::Int) => "i32".to_string(),
            Type::Int(IntTy::Byte) => "u8".to_string(),
            Type::Int(IntTy::UInt) => "usize".to_string(),
            Type::Float => "f64".to_string(),
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
                    Const(box AstNode::IntLiteral(size)) => size,
                    _ => todo!()
                };
                format!("[{}; {}]", self.transpile_type(inner_type), size)
            },
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
pub fn transpile(ast: &AstNode) -> String {
    let mut transpiler = Transpiler::new();
    transpiler.transpile(ast)
}
