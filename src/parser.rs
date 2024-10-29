//! Parser for Alloy
//!
//! This module is responsible for taking a stream of tokens from the lexer
//! and constructing an Abstract Syntax Tree (AST) that represents the structure
//! of an Alloy program.

use rand::seq::index;
use thin_vec::{thin_vec, ThinVec};
use tracing::{debug, error, info, instrument, trace};

use crate::ast::ty::{
    FnRetTy, Function, GenericParam, GenericParamKind, Ident, Param, Ty, TyKind, TypeOp,
};
use crate::ast::{
    AstNode, BinaryOperator, BindAttr, FnAttr, ImplKind, Precedence, UnaryOperator, WithClauseItem,
    P,
};
use crate::error::ParserError;
use crate::lexer::Token;
use itertools::{Itertools, MultiPeek};
use std::iter::Peekable;
use std::vec::IntoIter;

/// The Parser struct holds the state during parsing.
#[derive(Debug)]
pub struct Parser {
    tokens: Peekable<IntoIter<Token>>,
    last_node: Option<Box<AstNode>>,
}

impl Parser {
    /// Creates a new Parser instance.
    pub fn new(tokens: Vec<Token>) -> Self {
        Parser {
            tokens: tokens.into_iter().peekable(),
            last_node: None,
        }
    }

    /// Peeks at the next token without consuming it.
    fn peek(&mut self) -> Option<&Token> {
        self.tokens.peek()
    }

    /// Advances to the next token, consuming the current one.
    #[instrument]
    fn advance(&mut self) -> Option<Token> {
        let token = self.tokens.next();
        trace!("advance: {:?}", token);
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

    fn store_node(&mut self, node: Box<AstNode>) -> Box<AstNode> {
        self.last_node.replace(node.clone());
        node
    }

    /// Consumes the expected token or returns an error.
    #[instrument]
    fn consume(&mut self, expected: &Token) -> Result<(), ParserError> {
        if self.consume_if(expected) {
            trace!("consumed {:?}", expected);
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

    fn consume_any<'a>(&'a mut self, options: &'a [&'a Token]) -> Result<&'a Token, ParserError> {
        let next = self.peek();
        for test in options.iter() {
            if next == Some(test) {
                self.advance();
                return Ok(test);
            }
        }
        Err(ParserError::ExpectedToken(
            format!("one of {:?}", options),
            format!("{:?}", next),
        ))
    }

    fn consume_newlines(&mut self) {
        while self.consume_if(&Token::Newline) {}
        trace!("consumed newlines");
    }
    /// Parses the entire program.
    #[instrument(skip(self))]
    pub fn parse(&mut self) -> Result<Box<AstNode>, ParserError> {
        let mut declarations = ThinVec::new();
        while !self.is_at_end() {
            // Skip any leading newlines
            while self.consume_if(&Token::Newline) {}
            if !self.is_at_end() {
                match self.parse_declaration() {
                    Ok(decl) => {
                        match decl {
                            box AstNode::PipelineOperation { ref prev, .. } => {
                                // Special case for pipeline operations
                                // The pipeline is created as a separate node, but we want to
                                // merge it with the previous node, if that's the node it refers
                                if let Some(last_decl) = declarations.last_mut() {
                                    if last_decl == prev {
                                        *last_decl = decl;
                                    }
                                }
                            }
                            _ => declarations.push(decl),
                        }
                    }
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
    #[instrument]
    fn parse_primary(&mut self) -> Result<Box<AstNode>, ParserError> {
        trace!("parsing primary");
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
            Some(Token::Pipeline) => {
                if let Some(last_node) = self.last_node.clone() {
                    self.parse_pipeline(last_node)
                } else {
                    Err(ParserError::UnexpectedToken(
                        "Expected expression".to_string(),
                    ))
                }
            }
            Some(t) => {
                if let Some(op) = UnaryOperator::from_token(&t) {
                    self.parse_unary(op)
                } else {
                    Err(ParserError::UnexpectedToken(format!(
                        "in primary expression: {:?}",
                        t
                    )))
                }
            }
            t => {
                debug!("primary {:?}, tokens {:?}", t, self.tokens);
                Err(ParserError::UnexpectedToken(format!(
                    "in primary expression: {:?}",
                    t
                )))
            }
        }
    }

    /// Parses a declaration (function or variable).
    #[instrument(skip(self))]
    pub fn parse_declaration(&mut self) -> Result<Box<AstNode>, ParserError> {
        debug!("parsing declaration");
        let next = self.peek().map(Token::ident_to_keyword);
        debug!("next: {:?}", next);
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
            Some(Token::Pipeline) => {
                if let Some(last_node) = self.last_node.clone() {
                    self.parse_pipeline(last_node)
                } else {
                    Err(ParserError::UnexpectedToken(
                        "Expected expression".to_string(),
                    ))
                }
            }
            //Some(Token::Shared) => todo!(),
            _ => self.parse_statement(),
        }?;

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
        Ok(self.store_node(declaration))
    }

    /// Parses a function declaration.
    #[instrument(skip(self))]
    fn parse_function_declaration(&mut self) -> Result<Box<AstNode>, ParserError> {
        self.consume(&Token::Fn)?;
        self.finish_fn_declaration()
    }

    #[instrument]
    fn parse_struct_decl(&mut self) -> Result<Box<AstNode>, ParserError> {
        self.consume(&Token::Struct)?;
        let (name, generic_params) = self.parse_delc_start().inspect_err(|e| {
            error!(%e);
        })?;
        self.consume_newlines();
        let mut members = ThinVec::new();
        let next = self.peek();

        if next == Some(&Token::LParen) {
            debug!("parsing tuple struct");
            self.parse_tuple_struct_decl(name, generic_params)
        } else {
            debug!("parsing struct");
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
                members.push(self.parse_member(ImplKind::Struct).inspect_err(|e| {
                    error!(%e);
                })?);
            }
            Ok(P(AstNode::StructDeclaration {
                name,
                generic_params,
                where_clause: ThinVec::new(),
                members,
            }))
        }
    }

    #[instrument]
    fn parse_tuple_struct_decl(
        &mut self,
        name: Ident,
        generic_params: ThinVec<GenericParam>,
    ) -> Result<Box<AstNode>, ParserError> {
        let mut params = ThinVec::new();
        let mut index = 0;
        self.consume(&Token::LParen)?;
        self.consume_newlines();
        while !self.check(&Token::RParen) {
            // TODO: need to add visibility here and elsewhere
            let name = format!("{}", index);
            index += 1;
            let type_annotation = self.parse_type_annotation().inspect_err(|e| {
                error!(%e);
            })?;
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

    #[instrument]
    fn parse_enum_decl(&mut self) -> Result<Box<AstNode>, ParserError> {
        self.consume(&Token::Enum)?;
        let (name, generic_params) = self.parse_delc_start().inspect_err(|e| {
            error!(%e);
        })?;
        self.consume_newlines();
        let mut variants = ThinVec::new();
        self.consume(&Token::LBrace)?;
        loop {
            self.consume_newlines();
            if self.consume_if(&Token::RBrace) {
                break;
            }
            variants.push(self.parse_enum_variant().inspect_err(|e| {
                error!(%e);
            })?);
        }
        Ok(P(AstNode::EnumDeclaration {
            name,
            generic_params,
            where_clause: ThinVec::new(),
            variants,
        }))
    }

    #[instrument]
    fn parse_trait_decl(&mut self) -> Result<Box<AstNode>, ParserError> {
        self.consume(&Token::Trait)?;
        let (name, generic_params) = self.parse_delc_start().inspect_err(|e| {
            error!(%e);
        })?;
        self.consume_newlines();
        let mut bounds = None;
        if self.consume_if(&Token::Colon) || self.consume_if(&Token::Assign) {
            bounds = Some(self.parse_type_op().inspect_err(|e| {
                error!(%e);
            })?);
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
            members.push(self.parse_member(ImplKind::Trait).inspect_err(|e| {
                error!(%e);
            })?);
        }
        Ok(P(AstNode::TraitDeclaration {
            name,
            generic_params,
            bounds,
            where_clause: ThinVec::new(),
            members,
        }))
    }

    #[instrument]
    fn parse_union_decl(&mut self) -> Result<Box<AstNode>, ParserError> {
        self.consume(&Token::Union)?;
        let (name, generic_params) = self.parse_delc_start().inspect_err(|e| {
            error!(%e);
        })?;
        self.consume_newlines();
        todo!()
    }

    #[instrument]
    fn parse_effect_decl(&mut self) -> Result<Box<AstNode>, ParserError> {
        print!("parsing effect decl: ");
        self.consume(&Token::Effect)?;
        let (name, generic_params) = self.parse_delc_start().inspect_err(|e| {
            error!(%e);
        })?;
        println!("{:?}[{:?}]", name, generic_params);
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
            members.push(self.parse_effect_member().inspect_err(|e| {
                error!(%e);
            })?);
        }
        Ok(P(AstNode::EffectDeclaration {
            name,
            generic_params,
            bounds,
            where_clause: ThinVec::new(),
            members,
        }))
    }

    #[instrument]
    fn parse_handler_decl(&mut self) -> Result<Box<AstNode>, ParserError> {
        self.consume(&Token::Handler)?;
        // do we need additional generics here?
        let (name, generic_params) = self.parse_delc_start().inspect_err(|e| {
            error!(%e);
        })?;
        self.consume_newlines();
        self.consume(&Token::For)?;
        let (target, target_generic_params) = self.parse_delc_start().inspect_err(|e| {
            error!(%e);
        })?;
        self.consume_newlines();
        let mut bounds = None;
        if self.consume_if(&Token::Colon) {
            bounds = Some(self.parse_type_op().inspect_err(|e| {
                error!(%e);
            })?);
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
            members.push(self.parse_member(ImplKind::Handler).inspect_err(|e| {
                error!(%e);
            })?);
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
    #[instrument]
    fn parse_impl_decl(&mut self) -> Result<Box<AstNode>, ParserError> {
        self.consume(&Token::Impl)?;
        // do we need additional generics here?
        let (name, generic_params) = self.parse_delc_start().inspect_err(|e| {
            error!(%e);
        })?;
        self.consume_newlines();
        let kind = if self.consume_if(&Token::For) {
            ImplKind::Infer
        } else {
            ImplKind::Struct
        };

        let (target, target_generic_params) = if kind == ImplKind::Infer {
            self.parse_delc_start().inspect_err(|e| {
                error!(%e);
            })?
        } else {
            (String::new(), ThinVec::new())
        };
        self.consume_newlines();
        let mut bounds = None;
        if self.consume_if(&Token::Colon) {
            bounds = Some(self.parse_type_op().inspect_err(|e| {
                error!(%e);
            })?);
        }
        let mut members = ThinVec::new();
        if self.is_marker()? {
            return Ok(P(AstNode::ImplDeclaration {
                name,
                generic_params,
                kind,
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
            members.push(self.parse_member(kind).inspect_err(|e| {
                error!(%e);
            })?);
        }
        Ok(P(AstNode::ImplDeclaration {
            name,
            generic_params,
            kind,
            target,
            target_generic_params,
            where_clause: ThinVec::new(),
            bounds,
            members,
        }))
    }

    #[instrument(skip(self))]
    fn parse_member(&mut self, kind: ImplKind) -> Result<Box<AstNode>, ParserError> {
        let next = self.peek();
        match next {
            Some(Token::Fn) => self.parse_function_declaration(),
            Some(&Token::Let) if kind == ImplKind::Handler => self.parse_variable_declaration(),
            Some(&Token::Default) => todo!(),
            Some(&Token::Shared) => todo!(),
            Some(&Token::Type) => todo!(),
            _ => match kind {
                ImplKind::Struct => self.parse_struct_field(),
                ImplKind::Enum => self.parse_enum_variant(),
                _ => self.parse_statement(), // Maybe we should return an error here?
            },
        }
    }

    #[instrument(skip(self))]
    fn parse_effect_member(&mut self) -> Result<Box<AstNode>, ParserError> {
        self.finish_fn_declaration()
    }

    #[instrument(skip(self))]
    fn parse_enum_variant(&mut self) -> Result<Box<AstNode>, ParserError> {
        self.parse_struct_decl()
    }

    #[instrument(skip(self))]
    fn parse_struct_field(&mut self) -> Result<Box<AstNode>, ParserError> {
        self.finish_variable_declaration()
    }

    #[instrument]
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

    #[instrument(skip(self))]
    fn parse_delc_start(&mut self) -> Result<(Ident, ThinVec<GenericParam>), ParserError> {
        let name = self.parse_identifier().inspect_err(|e| {
            debug!(%e);
        })?;
        let generic_params = if self.consume_if(&Token::LBracket) {
            let params = self.parse_generic_params().inspect_err(|e| {
                error!(%e);
            })?;
            self.consume(&Token::RBracket)?;
            params
        } else {
            ThinVec::new()
        };
        Ok((name, generic_params))
    }

    #[instrument]
    fn finish_fn_declaration(&mut self) -> Result<Box<AstNode>, ParserError> {
        let name = self.parse_identifier().inspect_err(|e| {
            debug!(%e);
        })?;
        debug!("finishing fn declaration {:?}", name);
        let generic_params = if self.consume_if(&Token::LBracket) {
            let params = self.parse_generic_params().inspect_err(|e| {
                error!(%e);
            })?;
            self.consume(&Token::RBracket)?;
            params
        } else {
            ThinVec::new()
        };

        self.consume(&Token::LParen)?;
        let params = self.parse_parameters().inspect_err(|e| {
            error!(%e);
        })?;
        self.consume(&Token::RParen)?;

        let return_type = if self.consume_if(&Token::Arrow) {
            Some(self.parse_type_annotation()?)
        } else {
            None
        };
        let mut attrs = ThinVec::new();
        if self.peek() == Some(&Token::With) {
            let with_clause = self.parse_with_clause().inspect_err(|e| {
                error!(%e);
            })?;
            let is_async = with_clause.iter().any(|item| {
                matches!(
                    item.clone(),
                    box WithClauseItem::Generic(
                        GenericParam { name, kind: GenericParamKind::Type(None), .. }
                    ) if name == "async"
                )
            });
            attrs.push(FnAttr {
                is_async,
                is_shared: false,
                effects: with_clause,
            });
        }
        println!("{}({:?}) -> {:?}", name, params, return_type);
        let body = if self.peek() == Some(&Token::LBrace) {
            self.parse_block().inspect_err(|e| {
                error!(%e);
            })?
        } else {
            ThinVec::new()
        };
        Ok(P(AstNode::FunctionDeclaration {
            name,
            attrs,
            function: Function {
                generic_params,
                inputs: params,

                output: return_type.map(FnRetTy::Ty).unwrap_or_default(),
            },
            body,
        }))
    }

    #[instrument]
    fn parse_with_clause(&mut self) -> Result<ThinVec<Box<WithClauseItem>>, ParserError> {
        self.consume(&Token::With)?;
        let mut clauses = ThinVec::new();
        while !self.check(&Token::LBrace) && !self.check(&Token::Semicolon) {
            // currently only handles the simple cases
            clauses.push(P(WithClauseItem::Generic(GenericParam::simple(
                self.parse_identifier().inspect_err(|e| {
                    debug!(%e);
                })?,
            ))));
            if !self.consume_if(&Token::Comma) {
                break;
            }
        }
        Ok(clauses)
    }

    #[instrument]
    fn parse_function_call(&mut self, callee: Box<AstNode>) -> Result<Box<AstNode>, ParserError> {
        let arguments = self.parse_arguments().inspect_err(|e| {
            error!(%e);
        })?;
        if self.check(&Token::LBrace) && self.will_occur_in_next_scope(&Token::In) {
            self.consume(&Token::LBrace)?;
            self.parse_trailing_closure(callee).map(P).inspect_err(|e| {
                error!(%e);
            })
        } else {
            Ok(P(AstNode::FunctionCall { callee, arguments }))
        }
    }

    /// Parses a generic function call.
    /// Falls through to treating as an identifier if it wasn't a generic function call, mostly.
    #[instrument]
    fn parse_generic_function_call(&mut self, callee: String) -> Result<Box<AstNode>, ParserError> {
        let generic_args = if self.consume_if(&Token::LBracket) {
            let mut params = ThinVec::new();
            while !self.check(&Token::RBracket) {
                params.push(P(Ty::simple(self.parse_identifier().inspect_err(|e| {
                    debug!(%e);
                })?)));
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
        let arguments = self.parse_arguments().inspect_err(|e| {
            error!(%e);
        })?;
        if self.check(&Token::LBrace) && self.will_occur_in_next_scope(&Token::In) {
            self.consume(&Token::LBrace)?;
            self.parse_trailing_closure(P(AstNode::Identifier(callee)))
                .map(P)
                .inspect_err(|e| {
                    error!(%e);
                })
        } else {
            Ok(P(AstNode::GenericFunctionCall {
                name: callee,
                generic_args,
                arguments,
            }))
        }
    }

    // Helper method to parse generic parameters
    #[instrument]

    fn parse_generic_params(&mut self) -> Result<ThinVec<GenericParam>, ParserError> {
        let mut params = ThinVec::new();
        while !self.check(&Token::RBracket) {
            params.push(GenericParam::simple(self.parse_identifier().inspect_err(
                |e| {
                    debug!(%e);
                },
            )?));
            if !self.consume_if(&Token::Comma) {
                break;
            }
        }
        Ok(params)
    }

    // Helper method to parse function parameters
    #[instrument]

    fn parse_parameters(&mut self) -> Result<ThinVec<Param>, ParserError> {
        let mut params = ThinVec::new();
        if !self.check(&Token::RParen) {
            loop {
                let name = self.parse_identifier().inspect_err(|e| {
                    error!(%e);
                })?;
                let type_annotation = if name == "self" {
                    P(Ty {
                        kind: TyKind::SelfType,
                    })
                } else {
                    self.consume(&Token::Colon)?;
                    self.parse_type_annotation().inspect_err(|e| {
                        error!(%e);
                    })?
                };
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
    #[instrument]
    fn parse_type_annotation(&mut self) -> Result<Box<Ty>, ParserError> {
        if self.consume_if(&Token::Pipe) {
            let mut params = ThinVec::new();
            loop {
                if self.check(&Token::Pipe) {
                    break;
                }
                params.push(Param {
                    name: "".to_string(),
                    ty: self.parse_type_annotation().inspect_err(|e| {
                        error!(%e);
                    })?,
                });
                if !self.consume_if(&Token::Comma) {
                    break;
                }
            }
            self.consume(&Token::Pipe)?;
            let return_type = if self.consume_if(&Token::Arrow) {
                Some(self.parse_type_annotation().inspect_err(|e| {
                    error!(%e);
                })?)
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
                params.push(self.parse_type_annotation().inspect_err(|e| {
                    error!(%e);
                })?);
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
    #[instrument]
    fn parse_variable_declaration(&mut self) -> Result<Box<AstNode>, ParserError> {
        self.consume(&Token::Let)?;
        self.finish_variable_declaration().inspect_err(|e| {
            error!(%e);
        })
    }

    #[instrument]
    fn finish_variable_declaration(&mut self) -> Result<Box<AstNode>, ParserError> {
        let mutable = self.consume_if(&Token::Mut);
        let name = self.parse_identifier()?;
        let type_annotation = if self.consume_if(&Token::Colon) {
            Some(self.parse_type_annotation().inspect_err(|e| {
                error!(%e);
            })?)
        } else {
            None
        };
        let initializer = if self.consume_if(&Token::Assign) {
            Some(self.parse_expression(Precedence::None).inspect_err(|e| {
                error!(%e);
            })?)
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
    #[instrument]
    pub fn parse_statement(&mut self) -> Result<Box<AstNode>, ParserError> {
        // Skip any leading newlines
        self.consume_newlines();
        let node = match self.peek() {
            Some(Token::If) => self.parse_if_statement(),
            Some(Token::While) => self.parse_while_statement(),
            Some(Token::For) => self.parse_for_statement(),
            Some(Token::Guard) => self.parse_guard_statement(),
            Some(Token::Return) => self.parse_return_statement(),
            Some(Token::LBrace) => Ok(P(AstNode::Block(self.parse_block()?))),
            Some(Token::Let) => self.parse_variable_declaration(),
            Some(Token::Run) => todo!(),
            Some(Token::Pipeline) => {
                if let Some(last_node) = self.last_node.clone() {
                    self.parse_pipeline(last_node)
                } else {
                    Err(ParserError::UnexpectedToken(
                        "Expected expression".to_string(),
                    ))
                }
            }
            _ => self.parse_expression(Precedence::None).and_then(|expr| {
                self.consume_if(&Token::Semicolon);
                Ok(expr)
            }),
        };
        debug!(?node);
        node.map(|node| self.store_node(node))
    }

    /// Parses an if statement.
    #[instrument]
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
    #[instrument]
    fn parse_while_statement(&mut self) -> Result<Box<AstNode>, ParserError> {
        self.advance(); // Consume 'while'
        self.consume(&Token::LParen)?;
        let condition = self.parse_expression(Precedence::None).inspect_err(|e| {
            error!(%e);
        })?;
        self.consume(&Token::RParen)?;
        let body = self.parse_block().inspect_err(|e| {
            error!(%e);
        })?;
        Ok(P(AstNode::WhileLoop {
            condition,
            body: P(AstNode::Block(body)),
        }))
    }

    /// Parses a for statement.
    #[instrument]
    fn parse_for_statement(&mut self) -> Result<Box<AstNode>, ParserError> {
        self.consume(&Token::For)?;
        let item = self.parse_identifier().inspect_err(|e| {
            error!(%e);
        })?;
        self.consume(&Token::In)?;
        let iterable = self.parse_expression(Precedence::None).inspect_err(|e| {
            error!(%e);
        })?;
        let body = self.parse_block().inspect_err(|e| {
            error!(%e);
        })?;
        Ok(P(AstNode::ForInLoop {
            item,
            iterable,
            body: P(AstNode::Block(body)),
        }))
    }

    /// Parses a guard statement.
    #[instrument]
    fn parse_guard_statement(&mut self) -> Result<Box<AstNode>, ParserError> {
        self.consume(&Token::Guard)?;
        let condition = self.parse_expression(Precedence::None).inspect_err(|e| {
            error!(%e);
        })?;
        self.consume(&Token::Else)?;
        let body = self.parse_statement().inspect_err(|e| {
            error!(%e);
        })?;
        Ok(P(AstNode::GuardStatement { condition, body }))
    }

    /// Parses a return statement.
    #[instrument]
    fn parse_return_statement(&mut self) -> Result<Box<AstNode>, ParserError> {
        self.consume(&Token::Return)?;
        let value = if !self.check(&Token::Semicolon)
            && !self.check(&Token::RBrace)
            && !self.check(&Token::Newline)
        {
            println!("parsing expression");
            Some(self.parse_expression(Precedence::None).inspect_err(|e| {
                error!(%e);
            })?)
        } else {
            None
        };
        self.consume_if(&Token::Semicolon);
        Ok(P(AstNode::ReturnStatement(value)))
    }

    #[instrument]
    fn parse_statement_or_block(&mut self) -> Result<Box<AstNode>, ParserError> {
        if self.check(&Token::LBrace) {
            Ok(P(AstNode::Block(self.parse_block().inspect_err(|e| {
                error!(%e);
            })?)))
        } else {
            Ok(self.parse_statement().inspect_err(|e| {
                error!(%e);
            })?)
        }
    }

    /// Parses a block of statements.
    #[instrument]
    fn parse_block(&mut self) -> Result<ThinVec<Box<AstNode>>, ParserError> {
        self.consume(&Token::LBrace)?;

        let mut statements = ThinVec::new();
        while !self.check(&Token::RBrace) && !self.is_at_end() {
            // Skip any leading newlines
            self.consume_newlines();
            if self.check(&Token::RBrace) {
                break;
            }
            statements.push(self.parse_statement().inspect_err(|e| {
                error!(%e);
            })?);
        }
        debug!("{:?}", statements);

        // Skip any trailing newlines
        self.consume_newlines();

        self.consume(&Token::RBrace)?;
        Ok(statements)
    }

    #[instrument]
    fn parse_trailing_closure(&mut self, callee: Box<AstNode>) -> Result<AstNode, ParserError> {
        // Skip any leading newlines
        self.consume_newlines();
        let mut arguments = ThinVec::new();
        println!("parsing arguments");
        loop {
            if self.check(&Token::In) {
                break;
            }

            arguments.push(self.parse_expression(Precedence::None).inspect_err(|e| {
                error!(%e);
            })?);
            if !self.consume_if(&Token::Comma) {
                break;
            }
        }
        self.consume(&Token::In)?;
        println!("{:?}", self.tokens);
        let closure = self.finish_trailing_closure().inspect_err(|e| {
            error!(%e);
        })?;
        Ok(AstNode::TrailingClosure {
            callee: P(AstNode::FunctionCall { callee, arguments }),
            closure: P(closure),
        })
    }

    /// Finishes parsing a trailing closure.
    #[instrument]
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
            statements.push(self.parse_statement().inspect_err(|e| {
                error!(%e);
            })?);
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
    #[instrument]
    pub fn parse_expression(
        &mut self,
        precedence: Precedence,
    ) -> Result<Box<AstNode>, ParserError> {
        let mut left = self.parse_primary().inspect_err(|e| {
            error!(%e, ?precedence);
        })?;
        debug!(
            "parsed primary: {:?}, next precedence: {:?}, next token: {:?}",
            left,
            self.get_precedence(),
            self.peek()
        );
        if self.at_expression_end() {
            trace!("at expression end");
            return Ok(left);
        }
        while precedence < self.get_precedence() {
            if self.at_expression_end() {
                break;
            }
            left = self.parse_infix(left).inspect_err(|e| {
                error!(%e, ?precedence);
            })?;
        }
        trace!("parsed infix: {:?}", left);
        // Check for trailing closure
        if self.check(&Token::LBrace) && self.will_occur_in_next_scope(&Token::In) {
            println!("found closure");
            self.consume(&Token::LBrace)?;
            left = P(self.parse_trailing_closure(left).inspect_err(|e| {
                error!(%e, ?precedence);
            })?);
        }

        Ok(left)
    }

    /// Parses an array literal.
    #[instrument]
    fn parse_array_literal(&mut self) -> Result<Box<AstNode>, ParserError> {
        let mut elements = ThinVec::new();
        while !self.check(&Token::RBracket) {
            elements.push(self.parse_expression(Precedence::None).inspect_err(|e| {
                error!(%e);
            })?);

            if !self.consume_if(&Token::Comma) {
                break;
            }
        }
        debug!("{:?}", elements);
        self.consume(&Token::RBracket)?;
        Ok(P(AstNode::ArrayLiteral(elements)))
    }

    /// Parses an infix expression.
    #[instrument]
    fn parse_infix(&mut self, left: Box<AstNode>) -> Result<Box<AstNode>, ParserError> {
        let next = self.peek();
        let prec = next.map_or_else(|| Precedence::None, |t| Precedence::from_token(t));
        debug!(
            "left: {:?}, next precedence: {:?}, next token: {:?}",
            left, prec, next
        );
        match prec {
            Precedence::Unary => {
                let operator = self
                    .advance()
                    .map(|t| {
                        UnaryOperator::from_token(&t).ok_or(ParserError::UnexpectedToken(
                            "Expected unary operator".to_string(),
                        ))
                    })
                    .ok_or(ParserError::UnexpectedToken(
                        "Expected unary operator".to_string(),
                    ))
                    .inspect_err(|e| {
                        error!(%e, ?prec);
                    })??;
                let operand = self.parse_expression(Precedence::Unary).inspect_err(|e| {
                    error!(%e);
                })?;
                Ok(P(AstNode::UnaryOperation { operator, operand }))
            }
            Precedence::Term
            | Precedence::Factor
            | Precedence::Equality
            | Precedence::Or
            | Precedence::And
            | Precedence::Comparison => self.parse_binary(left, prec).inspect_err(|e| {
                error!(%e);
            }),
            Precedence::Assignment => self.parse_assignment(left).inspect_err(|e| {
                error!(%e);
            }),
            Precedence::Pipeline => self.parse_pipeline(left).inspect_err(|e| {
                error!(%e);
            }),
            Precedence::Call => self.parse_member_access(left).inspect_err(|e| {
                error!(%e);
            }),

            _ => Ok(left),
        }
    }

    #[instrument]
    fn parse_member_access(&mut self, parent: Box<AstNode>) -> Result<Box<AstNode>, ParserError> {
        self.consume(&Token::Dot)?;
        let child = self.parse_expression(Precedence::Call).inspect_err(|e| {
            error!(%e);
        })?;

        match child {
            box AstNode::FunctionCall { callee, arguments } => match (callee, parent) {
                (box AstNode::Identifier(name), box AstNode::Identifier(parent_name)) => {
                    let path = AstNode::Identifier(format!("{}.{}", parent_name, name));
                    Ok(P(AstNode::FunctionCall {
                        callee: P(path),
                        arguments,
                    }))
                }
                _ => todo!(),
            },
            box AstNode::GenericFunctionCall {
                name,
                generic_args,
                arguments,
            } => match parent {
                box AstNode::Identifier(parent_name) => {
                    let path = format!("{}.{}", parent_name, name);
                    Ok(P(AstNode::GenericFunctionCall {
                        name: path,
                        generic_args,
                        arguments,
                    }))
                }
                _ => todo!(),
            },
            box AstNode::Identifier(name) => todo!(),
            _ => Err(ParserError::ExpectedToken(
                "identifier".to_string(),
                format!("{:?}", child),
            )),
        }
    }

    #[instrument]
    fn parse_unary(&mut self, op: UnaryOperator) -> Result<Box<AstNode>, ParserError> {
        let operand = self.parse_expression(Precedence::Unary).inspect_err(|e| {
            error!(%e);
        })?;
        Ok(P(AstNode::UnaryOperation {
            operator: op,
            operand,
        }))
    }

    /// Parses an assignment operation.
    #[instrument]
    fn parse_assignment(&mut self, left: Box<AstNode>) -> Result<Box<AstNode>, ParserError> {
        self.advance(); // Consume the '=' token
        let value = self
            .parse_expression(Precedence::Assignment)
            .inspect_err(|e| {
                error!(%e);
            })?;
        Ok(P(AstNode::BinaryOperation {
            left,
            operator: BinaryOperator::Assign,
            right: value,
        }))
    }

    /// Parses a pipeline operation.
    #[instrument]
    fn parse_pipeline(&mut self, prev: Box<AstNode>) -> Result<Box<AstNode>, ParserError> {
        self.advance(); // Consume the '|>' token
        println!("parsing pipeline");
        let next = self
            .parse_expression(Precedence::Pipeline)
            .inspect_err(|e| {
                error!(%e);
            })?;
        Ok(P(AstNode::PipelineOperation { prev, next }))
    }

    /// Parses a binary operation.
    #[instrument]
    fn parse_binary(
        &mut self,
        left: Box<AstNode>,
        precedence: Precedence,
    ) -> Result<Box<AstNode>, ParserError> {
        let operator = self.advance().ok_or(ParserError::UnexpectedToken(
            "Expected binary operator".to_string(),
        ))?;
        let right = self.parse_expression(precedence).inspect_err(|e| {
            error!(%e);
        })?;
        Ok(P(AstNode::BinaryOperation {
            left,
            operator: self.token_to_binary_operator(operator).inspect_err(|e| {
                error!(%e);
            })?,
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

    /// Converts a token to a binary operator.
    fn token_to_unary_operator(&self, token: Token) -> Result<UnaryOperator, ParserError> {
        UnaryOperator::from_token(&token).ok_or(ParserError::UnexpectedToken(format!(
            "Unexpected token for unary operator: {:?}",
            token
        )))
    }

    /// Parses function arguments.
    #[instrument]
    fn parse_arguments(&mut self) -> Result<ThinVec<Box<AstNode>>, ParserError> {
        self.consume(&Token::LParen)?;
        let mut arguments = ThinVec::new();
        if !self.check(&Token::RParen) {
            loop {
                arguments.push(self.parse_expression(Precedence::None).inspect_err(|e| {
                    error!(%e);
                })?);
                debug!("parsed arguments: {:?}", arguments);
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
    #[instrument]
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
#[instrument]
pub fn parse(tokens: Vec<Token>) -> Result<Box<AstNode>, ParserError> {
    let mut parser = Parser::new(tokens);
    parser.parse()
}
