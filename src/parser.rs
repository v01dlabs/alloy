//! Parser for Alloy
//!
//! This module is responsible for taking a stream of tokens from the lexer
//! and constructing an Abstract Syntax Tree (AST) that represents the structure
//! of an Alloy program.

use thin_vec::{thin_vec, ThinVec};

use crate::ast::{AstNode, FnAttr, BinaryOperator, BindAttr, ImplKind, Precedence, P};
use crate::error::ParserError;
use crate::lexer::Token;
use crate::ast::ty::{ FnRetTy, Function, GenericParam, Ident, Param, Ty, TyKind, TypeOp};
use itertools::{Itertools, MultiPeek};
use std::iter::Peekable;
use std::vec::IntoIter;

/// The Parser struct holds the state during parsing.
pub struct Parser {
    tokens: Peekable<IntoIter<Token>>,
}

impl Parser {
    /// Creates a new Parser instance.
    pub fn new(tokens: Vec<Token>) -> Self {
        Parser {
            tokens: tokens.into_iter().peekable(),
        }
    }

    /// Peeks at the next token without consuming it.
    fn peek(&mut self) -> Option<&Token> {
        self.tokens.peek()
    }

    /// Advances to the next token, consuming the current one.
    fn advance(&mut self) -> Option<Token> {
        let token = self.tokens.next();
        println!("advance: {:?}", token);
        token
    }

    /// Checks if the next token matches the expected token.
    fn check(&mut self, expected: &Token) -> bool {
        if self.peek() == Some(expected) {
            true
        } else {
            false
        }
    }

    /// Consumes the next token if it matches the expected token.
    fn consume_if(&mut self, expected: &Token) -> bool {
        if self.check(expected) {
            self.advance();
            true
        } else {
            false
        }
    }

    /// Consumes the expected token or returns an error.
    fn consume(&mut self, expected: &Token) -> Result<(), ParserError> {
        if self.consume_if(expected) {
            println!("consumed {:?}", expected);
            Ok(())
        } else {
            Err(ParserError::ExpectedToken(
                format!("{:?}", expected),
                self.peek()
                    .map_or("end of input".to_string(), |t| format!("{:?}", t)),
            ))
        }
    }

    fn will_occur_in_scope(&mut self, expected: &Token) -> bool {
        let mut iter = self.tokens.clone().multipeek();
        while let Some(t) = iter.peek() {
            if t == expected {
                return true;
            }
            if t.is_block_end() || t.is_block_start() {
                return false;
            }
        }
        false
    }

    fn will_occur_in_next_scope(&mut self, expected: &Token) -> bool {
        let mut iter = self.tokens.clone().multipeek();
        while let Some(t) = iter.peek() {
            if t == expected {
                return true;
            }
            if t.is_block_end() || t.is_block_start() && t != &Token::LBrace {
                return false;
            }
        }
        false
    }

    /// Consumes the next token if it matches either of the given tokens.
    fn consume_either(&mut self, first: &Token, second: &Token) -> Result<(), ParserError> {
        if self.check(first) {
            self.consume(first)
        } else if self.check(second) {
            self.consume(second)
        } else {
            Err(ParserError::ExpectedToken(
                format!("{:?} or {:?}", first, second),
                self.peek()
                    .map_or("end of input".to_string(), |t| format!("{:?}", t)),
            ))
        }
    }

    /// Consumes the next token if it matches either of the given tokens.
    fn consume_any(
        &mut self,
        first: &Token,
        second: &Token,
        third: &Token,
    ) -> Result<(), ParserError> {
        if self.check(first) {
            self.consume(first)
        } else if self.check(second) {
            self.consume(second)
        } else if self.check(third) {
            self.consume(third)
        } else {
            Err(ParserError::ExpectedToken(
                format!("{:?} or {:?}", first, second),
                self.peek()
                    .map_or("end of input".to_string(), |t| format!("{:?}", t)),
            ))
        }
    }

    fn consume_newlines(&mut self) {
        while self.consume_if(&Token::Newline) {}
    }
    /// Parses the entire program.
    pub fn parse(&mut self) -> Result<Box<AstNode>, ParserError> {
        let mut declarations = ThinVec::new();
        while !self.is_at_end() {
            // Skip any leading newlines
            while self.consume_if(&Token::Newline) {}

            if !self.is_at_end() {
                match self.parse_declaration() {
                    Ok(decl) => declarations.push(decl),
                    Err(e) => return Err(e),
                }
            }

            // Skip any trailing newlines
            while self.consume_if(&Token::Newline) {}
        }
        if declarations.is_empty() {
            Err(ParserError::InvalidExpression)
        } else {
            Ok(P(AstNode::Program(declarations)))
        }
    }

    /// Parses a primary expression.
    fn parse_primary(&mut self) -> Result<Box<AstNode>, ParserError> {
        println!("parsing primary");
        match self.advance() {
            Some(Token::Identifier(name)) => {
                if self.check(&Token::LParen) {
                    self.parse_function_call(P(AstNode::Identifier(name)))
                } else if self.check(&Token::LBracket) {
                    self.parse_generic_function_call(name)
                } else {
                    Ok(P(AstNode::Identifier(name)))
                }
            }
            Some(Token::IntLiteral(value)) => Ok(P(AstNode::IntLiteral(value))),
            Some(Token::FloatLiteral(value)) => Ok(P(AstNode::FloatLiteral(value))),
            Some(Token::StringLiteral(value)) => Ok(P(AstNode::StringLiteral(value))),
            Some(Token::BoolLiteral(value)) => Ok(P(AstNode::BoolLiteral(value))),
            Some(Token::LParen) => {
                let expr = self.parse_expression(Precedence::None)?;
                self.consume(&Token::RParen)?;
                Ok(expr)
            }
            Some(Token::LBracket) => self.parse_array_literal(),
            t => {
                println!("primary {:?}, tokens {:?}", t, self.tokens);
                Err(ParserError::UnexpectedToken(format!(
                    "in primary expression: {:?}",
                    t
                )))
            }
        }
    }

    /// Parses a declaration (function or variable).
    pub fn parse_declaration(&mut self) -> Result<Box<AstNode>, ParserError> {
        let next = self.peek().map(Token::ident_to_keyword);
        let declaration = match next {
            Some(Token::Let) => self.parse_variable_declaration(),
            Some(Token::Fn) => self.parse_function_declaration(),
            Some(Token::Effect) => self.parse_effect_decl(),
            Some(Token::Struct) => self.parse_struct_decl(),
            Some(Token::Enum) => self.parse_enum_decl(),
            Some(Token::Union) => self.parse_union_decl(),
            Some(Token::Trait) => self.parse_trait_decl(),
            Some(Token::Handler) => self.parse_handler_decl(),
            Some(Token::Impl) => self.parse_impl_decl(),
            Some(Token::Shared) => todo!(),
            _ => self.parse_statement(),
        };

        // Check if the declaration is followed by a newline or EOF
        let next_token = self.peek();
        if !matches!(next_token, Some(&Token::Newline) | None | Some(&Token::Eof)) {
            return Err(ParserError::ExpectedToken(
                "newline".to_string(),
                format!("{:?}", next_token.unwrap_or(&Token::Eof)),
            ));
        }

        // Consume the newline if present
        self.consume_if(&Token::Newline);

        declaration
    }

    /// Parses a function declaration.
    fn parse_function_declaration(&mut self) -> Result<Box<AstNode>, ParserError> {
        self.consume(&Token::Fn)?;
        self.finish_fn_declaration()
    }

    fn parse_struct_decl(&mut self) -> Result<Box<AstNode>, ParserError> {
        let (name, generic_params) = self.parse_delc_start()?;
        self.consume_newlines(); 
        let mut members = ThinVec::new();
        let next = self.peek();
        
        if next == Some(&Token::LParen) {
            self.parse_tuple_struct_decl(name, generic_params)
        } else {
            if self.is_marker()? {
                return Ok(P(AstNode::StructDeclaration {
                    name,
                    generic_params,
                    where_clause: ThinVec::new(),
                    members,
                }));
            }
            self.consume(&Token::LBrace)?;
            loop {
                self.consume_newlines();
                if self.consume_if(&Token::RBrace) {
                    break;
                }
                members.push(self.parse_member(ImplKind::Struct)?);
            }
            Ok(P(AstNode::StructDeclaration {
                name,
                generic_params,
                where_clause: ThinVec::new(),
                members,
            }))
        }

    }

    fn parse_tuple_struct_decl(&mut self, name: Ident, generic_params: ThinVec<GenericParam>) -> Result<Box<AstNode>, ParserError> {
        let mut params = ThinVec::new();
        let mut index = 0;
        self.consume(&Token::LParen)?;
        self.consume_newlines();
        while !self.check(&Token::RParen) {
            // TODO: need to add visibility here and elsewhere
            let name = format!("{}", index);
            index += 1;
            let type_annotation = self.parse_type_annotation()?;
            params.push(P(AstNode::VariableDeclaration {
                name,
                attrs: ThinVec::new(),
                type_annotation: Some(type_annotation),
                initializer: None,
            }));
            if !self.consume_if(&Token::Comma) {
                break;
            }
        }
        Ok(P(AstNode::StructDeclaration {
            name,
            generic_params,
            where_clause: ThinVec::new(),
            members: params,
        }))
    }

    fn parse_enum_decl(&mut self) -> Result<Box<AstNode>, ParserError> {
        let (name, generic_params) = self.parse_delc_start()?;
        self.consume_newlines();
        let mut variants = ThinVec::new();
        self.consume(&Token::LBrace)?;
        loop {
            self.consume_newlines();
            if self.consume_if(&Token::RBrace) {
                break;
            }
            variants.push(self.parse_enum_variant()?);
        }
        Ok(P(AstNode::EnumDeclaration {
            name,
            generic_params,
            where_clause: ThinVec::new(),
            variants,
        }))
    }

    fn parse_trait_decl(&mut self) -> Result<Box<AstNode>, ParserError> {
        let (name, generic_params) = self.parse_delc_start()?;
        self.consume_newlines();
        let mut bounds = None;
        if self.consume_if(&Token::Colon) || self.consume_if(&Token::Assign) {
            bounds = Some(self.parse_type_op()?);
        }
        if self.is_marker()? {  
            return Ok(P(AstNode::TraitDeclaration {
                name,
                generic_params,
                bounds,
                where_clause: ThinVec::new(),
                members: ThinVec::new(),
            }));
        }
        let mut members = ThinVec::new();
        self.consume(&Token::LBrace)?;
        loop {
            self.consume_newlines();
            if self.consume_if(&Token::RBrace) {
                break;
            }
            members.push(self.parse_member(ImplKind::Trait)?);
        }
        Ok(P(AstNode::TraitDeclaration {
            name,
            generic_params,
            bounds,
            where_clause: ThinVec::new(),
            members,
        }))
    }

    fn parse_union_decl(&mut self) -> Result<Box<AstNode>, ParserError> {
        let (name, generic_params) = self.parse_delc_start()?;
        self.consume_newlines();
        todo!()
    }

    fn parse_effect_decl(&mut self) -> Result<Box<AstNode>, ParserError> {
        let (name, generic_params) = self.parse_delc_start()?;
        self.consume_newlines();
        let mut bounds = None;
        if self.consume_if(&Token::Colon) {
            bounds = Some(self.parse_type_op()?);
        }
        let mut members = ThinVec::new();
        if self.is_marker()? {  
            return Ok(P(AstNode::EffectDeclaration {
                name,
                generic_params,
                bounds,
                where_clause: ThinVec::new(),
                members,
            }));
        }
        self.consume(&Token::LBrace)?;
        loop {
            self.consume_newlines();
            if self.consume_if(&Token::RBrace) {
                break;
            }
            members.push(self.parse_effect_member()?);
        }
        Ok(P(AstNode::EffectDeclaration {
            name,
            generic_params,
            bounds,
            where_clause: ThinVec::new(),
            members,
        }))
    }

    fn parse_handler_decl(&mut self) -> Result<Box<AstNode>, ParserError> {
        // do we need additional generics here?
        let (name, generic_params) = self.parse_delc_start()?;
        self.consume_newlines();
        self.consume(&Token::For)?;
        let (target, target_generic_params) = self.parse_delc_start()?;
        self.consume_newlines();
        let mut bounds = None;
        if self.consume_if(&Token::Colon) {
            bounds = Some(self.parse_type_op()?);
        }
        let mut members = ThinVec::new();
        if self.is_marker()? {  
            return Ok(P(AstNode::ImplDeclaration {
                name,
                generic_params,
                kind: ImplKind::Handler,
                target,
                target_generic_params,
                where_clause: ThinVec::new(),
                bounds,
                members,
            }));
        }
        self.consume(&Token::LBrace)?;
        loop {
            self.consume_newlines();
            if self.consume_if(&Token::RBrace) {
                break;
            }
            members.push(self.parse_member(ImplKind::Handler)?);
        }
        Ok(P(AstNode::ImplDeclaration {
            name,
            generic_params,
            kind: ImplKind::Handler,
            target,
            target_generic_params,
            where_clause: ThinVec::new(),
            bounds,
            members,
        }))
    }

    /// Parses a declaration for an implementation of a trait, struct, enum, etc.
    /// We can't distinguish them at this stage, that will happen later
    fn parse_impl_decl(&mut self) -> Result<Box<AstNode>, ParserError> {
        // do we need additional generics here?
        let (name, generic_params) = self.parse_delc_start()?;
        self.consume_newlines();
        self.consume(&Token::For)?;
        let (target, target_generic_params) = self.parse_delc_start()?;
        self.consume_newlines();
        let mut bounds = None;
        if self.consume_if(&Token::Colon) {
            bounds = Some(self.parse_type_op()?);
        }
        let mut members = ThinVec::new();
        if self.is_marker()? {  
            return Ok(P(AstNode::ImplDeclaration {
                name,
                generic_params,
                kind: ImplKind::Infer,
                target,
                target_generic_params,
                where_clause: ThinVec::new(),
                bounds,
                members,
            }));
        }
        self.consume(&Token::LBrace)?;
        loop {
            self.consume_newlines();
            if self.consume_if(&Token::RBrace) {
                break;
            }
            members.push(self.parse_member(ImplKind::Infer)?);
        }
        Ok(P(AstNode::ImplDeclaration {
            name,
            generic_params,
            kind: ImplKind::Infer,
            target,
            target_generic_params,
            where_clause: ThinVec::new(),
            bounds,
            members,
        }))
    }

    fn parse_member(&mut self, kind: ImplKind) -> Result<Box<AstNode>, ParserError> {
        let next = self.peek();
        match next {
            Some(Token::Fn) => self.parse_function_declaration(),
            Some(&Token::Let) if kind == ImplKind::Handler => self.parse_variable_declaration(),
            Some(&Token::Default) => todo!(),
            Some(&Token::Shared) => todo!(),
            Some(&Token::Type) => todo!(),
            _ => match kind {
                ImplKind::Struct => self.parse_variable_declaration(),
                ImplKind::Enum => self.parse_enum_variant(),
                _ => self.parse_statement(), // Maybe we should return an error here?
            }
        }
    }

    

    fn parse_effect_member(&mut self) -> Result<Box<AstNode>, ParserError> {
        self.finish_fn_declaration()
    }

    fn parse_enum_variant(&mut self) -> Result<Box<AstNode>, ParserError> {
        self.parse_struct_decl()
    }

    fn parse_type_op(&mut self) -> Result<TypeOp, ParserError> {
        todo!()
    }

    fn is_marker(&mut self) -> Result<bool, ParserError> {
        let next = self.peek();
        if next == Some(&Token::LBrace) {
            Ok(false)
        } else if next == Some(&Token::Semicolon) {
            Ok(true)
        } else if next == Some(&Token::Newline) {
            self.consume_newlines();
            if self.peek() == Some(&Token::LBrace) {
                Ok(false)
            } else {
                Ok(true)
            }
        } else {
            Err(ParserError::ExpectedToken(
                "newline, LBrace, or Semicolon".to_string(),
                self.peek()
                    .map_or("end of input".to_string(), |t| format!("{:?}", t)),
            ))
        }
    }

    fn parse_delc_start(&mut self) -> Result<(Ident, ThinVec<GenericParam>), ParserError> {
        let name = self.parse_identifier()?;
        let generic_params = if self.consume_if(&Token::LBracket) {
            let params = self.parse_generic_params()?;
            self.consume(&Token::RBracket)?;
            params
        } else {
            ThinVec::new()
        };
        Ok((name, generic_params))
    }

    fn finish_fn_declaration(&mut self) -> Result<Box<AstNode>, ParserError> {
        let name = self.parse_identifier()?;

        let generic_params = if self.consume_if(&Token::LBracket) {
            let params = self.parse_generic_params()?;
            self.consume(&Token::RBracket)?;
            params
        } else {
            ThinVec::new()
        };

        self.consume(&Token::LParen)?;
        let params = self.parse_parameters()?;
        self.consume(&Token::RParen)?;

        let return_type = if self.consume_if(&Token::Arrow) {
            Some(self.parse_type_annotation()?)
        } else {
            None
        };
        println!("{}({:?}) -> {:?}", name, params, return_type);
        let body = self.parse_block()?;

        Ok(P(AstNode::FunctionDeclaration {
            name,
            attrs: ThinVec::new(),  
            function: Function {
                generic_params,
                inputs: params,
                
                output: return_type.map(FnRetTy::Ty).unwrap_or_default(),
            },
            body,
        }))
    }

    fn parse_function_call(&mut self, callee: Box<AstNode>) -> Result<Box<AstNode>, ParserError> {
        let arguments = self.parse_arguments()?;
        if self.check(&Token::LBrace) && self.will_occur_in_next_scope(&Token::In) {
            self.consume(&Token::LBrace)?;
            self.parse_trailing_closure(callee).map(P)
        } else {
            Ok(P(AstNode::FunctionCall { callee, arguments }))
        }
    }

    /// Parses a generic function call.
    /// Falls through to treating as an identifier if it wasn't a generic function call, mostly.
    fn parse_generic_function_call(&mut self, callee: String) -> Result<Box<AstNode>, ParserError> {
        let generic_args = if self.consume_if(&Token::LBracket) {
            let mut params = ThinVec::new();
            while !self.check(&Token::RBracket) {
                params.push(P(Ty::simple(self.parse_identifier()?)));
                if !self.consume_if(&Token::Comma) {
                    break;
                }
            }
            self.consume(&Token::RBracket)?;
            params
        } else {
            return Ok(P(AstNode::Identifier(callee)));
        };
        if !self.check(&Token::LParen) {
            return Ok(P(AstNode::Identifier(callee)));
        }
        let arguments = self.parse_arguments()?;
        if self.check(&Token::LBrace) && self.will_occur_in_next_scope(&Token::In) {
            self.consume(&Token::LBrace)?;
            self.parse_trailing_closure(P(AstNode::Identifier(callee)))
                .map(P)
        } else {
            Ok(P(AstNode::GenericFunctionCall {
                name: callee,
                generic_args,
                arguments,
            }))
        }
    }

    // Helper method to parse generic parameters
    fn parse_generic_params(&mut self) -> Result<ThinVec<GenericParam>, ParserError> {
        let mut params = ThinVec::new();
        while !self.check(&Token::RBracket) {
            params.push(GenericParam::simple(self.parse_identifier()?));
            if !self.consume_if(&Token::Comma) {
                break;
            }
        }
        Ok(params)
    }

    // Helper method to parse function parameters
    fn parse_parameters(&mut self) -> Result<ThinVec<Param>, ParserError> {
        let mut params = ThinVec::new();
        if !self.check(&Token::RParen) {
            loop {
                let name = self.parse_identifier()?;
                self.consume(&Token::Colon)?;
                let type_annotation = self.parse_type_annotation()?;
                params.push(Param {
                    name,
                    ty: type_annotation,
                });
                if !self.consume_if(&Token::Comma) {
                    break;
                }
            }
        }
        Ok(params)
    }

    /// Parses type annotations.
    fn parse_type_annotation(&mut self) -> Result<Box<Ty>, ParserError> {
        if self.consume_if(&Token::Pipe) {
            let mut params = ThinVec::new();
            loop {
                if self.check(&Token::Pipe) {
                    break;
                }
                params.push(Param {
                    name: "".to_string(),
                    ty: self.parse_type_annotation()?,
                });
                if !self.consume_if(&Token::Comma) {
                    break;
                }
            }
            self.consume(&Token::Pipe)?;
            let return_type = if self.consume_if(&Token::Arrow) {
                Some(self.parse_type_annotation()?)
            } else {
                None
            };
            return Ok(P(Ty {
                kind: TyKind::Function(Function {
                    generic_params: ThinVec::new(),
                    inputs: params,
                    output: return_type.map(FnRetTy::Ty).unwrap_or_default(),
                }),
            }));
        }
        let base_type = self.parse_identifier()?;
        if self.consume_if(&Token::LBracket) {
            let mut params = ThinVec::new();
            loop {
                params.push(self.parse_type_annotation()?);
                if !self.consume_if(&Token::Comma) {
                    break;
                }
            }
            self.consume(&Token::RBracket)?;
            Ok(P(Ty {
                kind: TyKind::Generic(base_type, params),
            }))
        } else {
            Ok(P(Ty {
                kind: TyKind::Simple(base_type),
            }))
        }
    }

    /// Parses a variable declaration.
    fn parse_variable_declaration(&mut self) -> Result<Box<AstNode>, ParserError> {
        self.consume(&Token::Let)?;
        let mutable = self.consume_if(&Token::Mut);
        let name = self.parse_identifier()?;
        let type_annotation = if self.consume_if(&Token::Colon) {
            Some(self.parse_type_annotation()?)
        } else {
            None
        };
        let initializer = if self.consume_if(&Token::Assign) {
            Some(self.parse_expression(Precedence::None)?)
        } else {
            None
        };
        // Consume the semicolon if present, but don't require it
        self.consume_if(&Token::Semicolon);
        Ok(P(AstNode::VariableDeclaration {
            name,
            attrs: thin_vec![BindAttr::new(mutable, None)],
            type_annotation,
            initializer,
        }))
    }

    /// Parses a statement.
    pub fn parse_statement(&mut self) -> Result<Box<AstNode>, ParserError> {
        // Skip any leading newlines
        self.consume_newlines();
        match self.peek() {
            Some(Token::If) => self.parse_if_statement(),
            Some(Token::While) => self.parse_while_statement(),
            Some(Token::For) => self.parse_for_statement(),
            Some(Token::Guard) => self.parse_guard_statement(),
            Some(Token::Return) => self.parse_return_statement(),
            Some(Token::LBrace) => Ok(P(AstNode::Block(self.parse_block()?))),
            Some(Token::Let) => self.parse_variable_declaration(),
            _ => self.parse_expression(Precedence::None).and_then(|expr| {
                self.consume_if(&Token::Semicolon);
                Ok(expr)
            }),
        }
    }

    /// Parses an if statement.
    fn parse_if_statement(&mut self) -> Result<Box<AstNode>, ParserError> {
        self.advance(); // Consume 'if'
        let paren = self.consume_if(&Token::LParen);
        let condition = self.parse_expression(Precedence::None)?;
        if paren {
            self.consume(&Token::RParen)?;
        }
        let then_branch = P(AstNode::Block(self.parse_block()?));
        let else_branch = if self.consume_if(&Token::Else) {
            Some(P(AstNode::Block(self.parse_block()?)))
        } else {
            None
        };
        Ok(P(AstNode::IfStatement {
            condition,
            then_branch,
            else_branch,
        }))
    }

    /// Parses a while statement.
    fn parse_while_statement(&mut self) -> Result<Box<AstNode>, ParserError> {
        self.advance(); // Consume 'while'
        self.consume(&Token::LParen)?;
        let condition = self.parse_expression(Precedence::None)?;
        self.consume(&Token::RParen)?;
        let body = self.parse_block()?;
        Ok(P(AstNode::WhileLoop {
            condition,
            body: P(AstNode::Block(body)),
        }))
    }

    /// Parses a for statement.
    fn parse_for_statement(&mut self) -> Result<Box<AstNode>, ParserError> {
        self.consume(&Token::For)?;
        let item = self.parse_identifier()?;
        self.consume(&Token::In)?;
        let iterable = self.parse_expression(Precedence::None)?;
        let body = self.parse_block()?;
        Ok(P(AstNode::ForInLoop {
            item,
            iterable,
            body: P(AstNode::Block(body)),
        }))
    }

    /// Parses a guard statement.
    fn parse_guard_statement(&mut self) -> Result<Box<AstNode>, ParserError> {
        self.consume(&Token::Guard)?;
        let condition = self.parse_expression(Precedence::None)?;
        self.consume(&Token::Else)?;
        let body = self.parse_statement()?;
        Ok(P(AstNode::GuardStatement { condition, body }))
    }

    /// Parses a return statement.
    fn parse_return_statement(&mut self) -> Result<Box<AstNode>, ParserError> {
        self.consume(&Token::Return)?;
        let value = if !self.check(&Token::Semicolon)
            && !self.check(&Token::RBrace)
            && !self.check(&Token::Newline)
        {
            println!("parsing expression");
            Some(self.parse_expression(Precedence::None)?)
        } else {
            None
        };
        self.consume_if(&Token::Semicolon);
        Ok(P(AstNode::ReturnStatement(value)))
    }

    fn parse_statement_or_block(&mut self) -> Result<Box<AstNode>, ParserError> {
        if self.check(&Token::LBrace) {
            Ok(P(AstNode::Block(self.parse_block()?)))
        } else {
            Ok(self.parse_statement()?)
        }
    }

    /// Parses a block of statements.
    fn parse_block(&mut self) -> Result<ThinVec<Box<AstNode>>, ParserError> {
        self.consume(&Token::LBrace)?;

        let mut statements = ThinVec::new();
        while !self.check(&Token::RBrace) && !self.is_at_end() {
            // Skip any leading newlines
            self.consume_newlines();
            if self.check(&Token::RBrace) {
                break;
            }
            statements.push(self.parse_statement()?);
        }
        println!("{:?}", statements);

        // Skip any trailing newlines
        self.consume_newlines();

        self.consume(&Token::RBrace)?;
        Ok(statements)
    }

    fn parse_trailing_closure(&mut self, callee: Box<AstNode>) -> Result<AstNode, ParserError> {
        // Skip any leading newlines
        self.consume_newlines();
        let mut arguments = ThinVec::new();
        println!("parsing arguments");
        loop {
            if self.check(&Token::In) {
                break;
            }

            arguments.push(self.parse_expression(Precedence::None)?);
            if !self.consume_if(&Token::Comma) {
                break;
            }
        }
        self.consume(&Token::In)?;
        println!("{:?}", self.tokens);
        let closure = self.finish_trailing_closure()?;
        Ok(AstNode::TrailingClosure {
            callee: P(AstNode::FunctionCall { callee, arguments }),
            closure: P(closure),
        })
    }

    /// Finishes parsing a trailing closure.
    fn finish_trailing_closure(&mut self) -> Result<AstNode, ParserError> {
        let mut statements = ThinVec::new();
        // Skip any leading newlines
        self.consume_newlines();
        while !self.check(&Token::RBrace) && !self.is_at_end() {
            // Skip any leading newlines
            self.consume_newlines();
            if self.check(&Token::RBrace) {
                break;
            }
            statements.push(self.parse_statement()?);
        }
        println!("{:?}", statements);

        // Skip any trailing newlines
        self.consume_newlines();

        self.consume(&Token::RBrace)?;
        Ok(AstNode::Block(statements))
    }

    fn at_expression_end(&mut self) -> bool {
        self.check(&Token::Semicolon)
            || self.check(&Token::Newline)
            || self.check(&Token::RBrace)
            || self.is_at_end()
    }

    /// Parses an expression.
    pub fn parse_expression(
        &mut self,
        precedence: Precedence,
    ) -> Result<Box<AstNode>, ParserError> {
        let mut left = self.parse_primary()?;
        if self.at_expression_end() {
            return Ok(left);
        }
        while precedence < self.get_precedence() {
            if self.at_expression_end() {
                break;
            }
            left = self.parse_infix(left)?;
        }

        // Check for trailing closure
        if self.check(&Token::LBrace) && self.will_occur_in_next_scope(&Token::In) {
            println!("found closure");
            self.consume(&Token::LBrace)?;
            left = P(self.parse_trailing_closure(left)?);
        }

        Ok(left)
    }

    /// Parses an array literal.
    fn parse_array_literal(&mut self) -> Result<Box<AstNode>, ParserError> {
        let mut elements = ThinVec::new();
        while !self.check(&Token::RBracket) {
            elements.push(self.parse_expression(Precedence::None)?);

            if !self.consume_if(&Token::Comma) {
                break;
            }
        }
        println!("{:?}", elements);
        self.consume(&Token::RBracket)?;
        Ok(P(AstNode::ArrayLiteral(elements)))
    }

    /// Parses an infix expression.
    fn parse_infix(&mut self, left: Box<AstNode>) -> Result<Box<AstNode>, ParserError> {
        let prec = self
            .peek()
            .map_or_else(|| Precedence::None, |t| Precedence::from_token(t));
        match prec {
            Precedence::Term
            | Precedence::Factor
            | Precedence::Equality
            | Precedence::Or
            | Precedence::And
            | Precedence::Comparison => self.parse_binary(left, prec),
            Precedence::Assignment => self.parse_assignment(left),
            Precedence::Pipeline => self.parse_pipeline(left),
            Precedence::Call => self.parse_function_call(left),
            _ => Ok(left),
        }
    }

    /// Parses an assignment operation.
    fn parse_assignment(&mut self, left: Box<AstNode>) -> Result<Box<AstNode>, ParserError> {
        self.advance(); // Consume the '=' token
        let value = self.parse_expression(Precedence::Assignment)?;
        Ok(P(AstNode::BinaryOperation {
            left,
            operator: BinaryOperator::Assign,
            right: value,
        }))
    }

    /// Parses a pipeline operation.
    fn parse_pipeline(&mut self, left: Box<AstNode>) -> Result<Box<AstNode>, ParserError> {
        self.advance(); // Consume the '|>' token
        let right = self.parse_expression(Precedence::Pipeline)?;
        Ok(P(AstNode::PipelineOperation { left, right }))
    }

    /// Parses a binary operation.
    fn parse_binary(
        &mut self,
        left: Box<AstNode>,
        precedence: Precedence,
    ) -> Result<Box<AstNode>, ParserError> {
        let operator = self.advance().ok_or(ParserError::UnexpectedToken(
            "Expected binary operator".to_string(),
        ))?;
        let right = self.parse_expression(precedence)?;
        Ok(P(AstNode::BinaryOperation {
            left,
            operator: self.token_to_binary_operator(operator)?,
            right,
        }))
    }

    /// Converts a token to a binary operator.
    fn token_to_binary_operator(&self, token: Token) -> Result<BinaryOperator, ParserError> {
        BinaryOperator::from_token(&token).ok_or(ParserError::UnexpectedToken(format!(
            "Unexpected token for binary operator: {:?}",
            token
        )))
    }

    /// Parses function arguments.
    fn parse_arguments(&mut self) -> Result<ThinVec<Box<AstNode>>, ParserError> {
        self.consume(&Token::LParen)?;
        let mut arguments = ThinVec::new();
        if !self.check(&Token::RParen) {
            loop {
                arguments.push(self.parse_expression(Precedence::None)?);
                if !self.consume_if(&Token::Comma) {
                    break;
                }
            }
        }
        self.consume(&Token::RParen)?;
        Ok(arguments)
    }

    /// Gets the precedence of the current token.
    fn get_precedence(&mut self) -> Precedence {
        match self.peek() {
            Some(tok) => Precedence::from_token(tok),
            _ => Precedence::None,
        }
    }

    /// Parses an identifier.
    fn parse_identifier(&mut self) -> Result<Ident, ParserError> {
        match self.advance() {
            Some(Token::Identifier(name)) => Ok(name),
            _ => Err(ParserError::ExpectedToken(
                "identifier".to_string(),
                format!("{:?}", self.peek()),
            )),
        }
    }

    

    /// Checks if the parser has reached the end of the token stream.
    fn is_at_end(&mut self) -> bool {
        self.peek() == Some(&Token::Eof) || self.peek().is_none()
    }
}

/// Parses a vector of tokens into an AST.
pub fn parse(tokens: Vec<Token>) -> Result<Box<AstNode>, ParserError> {
    let mut parser = Parser::new(tokens);
    parser.parse()
}
