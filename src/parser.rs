//! Parser for Alloy
//!
//! This module is responsible for taking a stream of tokens from the lexer
//! and constructing an Abstract Syntax Tree (AST) that represents the structure
//! of an Alloy program.

use thin_vec::ThinVec;

use crate::ast::{AstNode, BinaryOperator, Precedence};
use crate::error::ParserError;
use crate::lexer::Token;
use crate::ty::{FnRetTy, Function, GenericParam, Ident, Param, Ty, TyKind};
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
        self.peek() == Some(expected)
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
            Ok(())
        } else {
            Err(ParserError::ExpectedToken(
                format!("{:?}", expected),
                self.peek()
                    .map_or("end of input".to_string(), |t| format!("{:?}", t)),
            ))
        }
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
            Ok(Box::new(AstNode::Program(declarations)))
        }
    }

    /// Parses a primary expression.
    fn parse_primary(&mut self) -> Result<Box<AstNode>, ParserError> {
        match self.advance() {
            Some(Token::Identifier(name)) => {
                if self.check(&Token::LParen) {
                    self.parse_function_call(Box::new(AstNode::Identifier(name)))
                } else {
                    Ok(Box::new(AstNode::Identifier(name)))
                }
            }
            Some(Token::IntLiteral(value)) => Ok(Box::new(AstNode::IntLiteral(value))),
            Some(Token::FloatLiteral(value)) => Ok(Box::new(AstNode::FloatLiteral(value))),
            Some(Token::StringLiteral(value)) => Ok(Box::new(AstNode::StringLiteral(value))),
            Some(Token::BoolLiteral(value)) => Ok(Box::new(AstNode::BoolLiteral(value))),
            Some(Token::LParen) => {
                let expr = self.parse_expression(Precedence::None)?;
                self.consume(&Token::RParen)?;
                Ok(expr)
            }
            Some(Token::LBracket) => self.parse_array_literal(),
            t => Err(ParserError::UnexpectedToken(format!(
                "in primary expression: {:?}",
                t
            ))),
        }
    }

    /// Parses a declaration (function or variable).
    pub fn parse_declaration(&mut self) -> Result<Box<AstNode>, ParserError> {
        let declaration = match self.peek() {
            Some(Token::Let) => self.parse_variable_declaration(),
            Some(Token::Fn) => self.parse_function_declaration(),
            _ => self.parse_statement(),
        };

        // Check if the declaration is followed by a newline or EOF
        if !matches!(
            self.peek(),
            Some(&Token::Newline) | None | Some(&Token::Eof)
        ) {
            return Err(ParserError::ExpectedToken(
                "newline".to_string(),
                format!("{:?}", self.peek().unwrap_or(&Token::Eof)),
            ));
        }

        // Consume the newline if present
        self.consume_if(&Token::Newline);

        declaration
    }

    /// Parses a function declaration.
    fn parse_function_declaration(&mut self) -> Result<Box<AstNode>, ParserError> {
        self.consume(&Token::Fn)?;
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

        Ok(Box::new(AstNode::FunctionDeclaration {
            name,
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
        if self.check(&Token::LBrace) {
            let closure = self.finish_trailing_closure()?;
            Ok(Box::new(AstNode::TrailingClosure {
                callee: Box::new(AstNode::FunctionCall {
                    callee,
                    arguments,
                }),
                closure: Box::new(closure),
            }))
        } else {
            Ok(Box::new(AstNode::FunctionCall {
                callee,
                arguments,
            }))
        }
    }

    /// Finishes parsing a trailing closure.
    fn finish_trailing_closure(&mut self) -> Result<AstNode, ParserError> {
        let body = self.parse_block()?;
        Ok(AstNode::Block(body))
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
                params.push(Param { name, ty: type_annotation });
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
                    ty: self.parse_type_annotation()?
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
            return Ok(Box::new(Ty {
                kind: TyKind::Function(Function { 
                    generic_params: ThinVec::new(),
                    inputs: params, 
                    output: return_type.map(
                        FnRetTy::Ty
                    ).unwrap_or_default()
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
            Ok(Box::new(Ty {
                kind: TyKind::Generic(base_type, params),
            }))
        } else {
            Ok(Box::new(Ty {
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
        Ok(Box::new(AstNode::VariableDeclaration {
            name,
            mutable,
            type_annotation,
            initializer,
        }))
    }

    /// Parses a statement.
    pub fn parse_statement(&mut self) -> Result<Box<AstNode>, ParserError> {
        // Skip any leading newlines
        while self.consume_if(&Token::Newline) {}
        match self.peek() {
            Some(Token::If) => self.parse_if_statement(),
            Some(Token::While) => self.parse_while_statement(),
            Some(Token::For) => self.parse_for_statement(),
            Some(Token::Guard) => self.parse_guard_statement(),
            Some(Token::Return) => self.parse_return_statement(),
            Some(Token::LBrace) => Ok(Box::new(AstNode::Block(self.parse_block()?))),
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
        let then_branch = self.parse_statement_or_block()?;
        let else_branch = if self.consume_if(&Token::Else) {
            Some(self.parse_statement_or_block()?)
        } else {
            None
        };
        Ok(Box::new(AstNode::IfStatement {
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
        Ok(Box::new(AstNode::WhileLoop {
            condition,
            body: Box::new(AstNode::Block(body)),
        }))
    }

    /// Parses a for statement.
    fn parse_for_statement(&mut self) -> Result<Box<AstNode>, ParserError> {
        self.consume(&Token::For)?;
        let item = self.parse_identifier()?;
        self.consume(&Token::In)?;
        let iterable = self.parse_expression(Precedence::None)?;
        let body = self.parse_block()?;
        Ok(Box::new(AstNode::ForInLoop {
            item,
            iterable,
            body: Box::new(AstNode::Block(body)),
        }))
    }

    /// Parses a guard statement.
    fn parse_guard_statement(&mut self) -> Result<Box<AstNode>, ParserError> {
        self.consume(&Token::Guard)?;
        let condition = self.parse_expression(Precedence::None)?;
        self.consume(&Token::Else)?;
        let body = self.parse_statement()?;
        Ok(Box::new(AstNode::GuardStatement { condition, body }))
    }

    /// Parses a return statement.
    fn parse_return_statement(&mut self) -> Result<Box<AstNode>, ParserError> {
        self.advance(); // Consume 'return'
        let value = if !self.check(&Token::Semicolon) && !self.check(&Token::RBrace) {
            Some(self.parse_expression(Precedence::None)?)
        } else {
            None
        };
        self.consume_if(&Token::Semicolon);
        Ok(Box::new(AstNode::ReturnStatement(value)))
    }

    fn parse_statement_or_block(&mut self) -> Result<Box<AstNode>, ParserError> {
        if self.check(&Token::LBrace) {
            Ok(Box::new(AstNode::Block(self.parse_block()?)))
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
            while self.consume_if(&Token::Newline) {}
            if self.check(&Token::RBrace) {
                break;
            }
            statements.push(self.parse_statement()?);
        }
        println!("{:?}", statements);

        // Skip any trailing newlines
        while self.consume_if(&Token::Newline) {}

        self.consume(&Token::RBrace)?;
        Ok(statements)
    }

    fn parse_trailing_closure(&mut self, callee: Box<AstNode>) -> Result<AstNode, ParserError> {
        self.consume(&Token::Pipe)?;
        let mut arguments = ThinVec::new();
        if !self.check(&Token::Pipe) {
            loop {
                arguments.push(self.parse_expression(Precedence::None)?);
                if !self.consume_if(&Token::Comma) {
                    break;
                }
            }
        }
        self.consume(&Token::Pipe)?;
        let closure = self.finish_trailing_closure()?;
        Ok(AstNode::TrailingClosure {
            callee: Box::new(AstNode::FunctionCall {
                callee,
                arguments,
            }),
            closure: Box::new(closure),
        })
    }

    /// Parses an expression.
    pub fn parse_expression(&mut self, precedence: Precedence) -> Result<Box<AstNode>, ParserError> {
        let mut left = self.parse_primary()?;

        while precedence < self.get_precedence() {
            if self.check(&Token::Semicolon) || self.is_at_end() {
                break;
            }
            left = self.parse_infix(left)?;
        }

        // Check for trailing closure
        if self.check(&Token::Pipe) {
            left = Box::new(self.parse_trailing_closure(left)?);
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
        Ok(Box::new(AstNode::ArrayLiteral(elements)))
    }

    /// Parses an infix expression.
    fn parse_infix(&mut self, left: Box<AstNode>) -> Result<Box<AstNode>, ParserError> {
        match self.peek() {
            Some(Token::Plus) | Some(Token::Minus) => self.parse_binary(left, Precedence::Term),
            Some(Token::Multiply) | Some(Token::Divide) => {
                self.parse_binary(left, Precedence::Factor)
            }
            Some(Token::Eq) | Some(Token::NotEq) => self.parse_binary(left, Precedence::Equality),
            Some(Token::Lt) | Some(Token::Gt) | Some(Token::LtEq) | Some(Token::GtEq) => {
                self.parse_binary(left, Precedence::Comparison)
            }
            Some(Token::And) => self.parse_binary(left, Precedence::And),
            Some(Token::Or) => self.parse_binary(left, Precedence::Or),
            Some(Token::Assign) => self.parse_assignment(left),
            Some(Token::Pipeline) => self.parse_pipeline(left),
            Some(Token::LParen) => self.parse_function_call(left),
            _ => Ok(left),
        }
    }

    /// Parses an assignment operation.
    fn parse_assignment(&mut self, left: Box<AstNode>) -> Result<Box<AstNode>, ParserError> {
        self.advance(); // Consume the '=' token
        let value = self.parse_expression(Precedence::Assignment)?;
        Ok(Box::new(AstNode::BinaryOperation {
            left,
            operator: BinaryOperator::Assign,
            right: value,
        }))
    }

    /// Parses a pipeline operation.
    fn parse_pipeline(&mut self, left: Box<AstNode>) -> Result<Box<AstNode>, ParserError> {
        self.advance(); // Consume the '|>' token
        let right = self.parse_expression(Precedence::Pipeline)?;
        Ok(Box::new(AstNode::PipelineOperation {
            left: left,
            right: right,
        }))
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
        Ok(Box::new(AstNode::BinaryOperation {
            left: left,
            operator: self.token_to_binary_operator(operator)?,
            right: right,   
        }))
    }

    /// Converts a token to a binary operator.
    fn token_to_binary_operator(&self, token: Token) -> Result<BinaryOperator, ParserError> {
        match token {
            Token::Plus => Ok(BinaryOperator::Add),
            Token::Minus => Ok(BinaryOperator::Subtract),
            Token::Multiply => Ok(BinaryOperator::Multiply),
            Token::Divide => Ok(BinaryOperator::Divide),
            Token::Eq => Ok(BinaryOperator::Equal),
            Token::NotEq => Ok(BinaryOperator::NotEqual),
            Token::Lt => Ok(BinaryOperator::LessThan),
            Token::Gt => Ok(BinaryOperator::GreaterThan),
            Token::LtEq => Ok(BinaryOperator::LessThanOrEqual),
            Token::GtEq => Ok(BinaryOperator::GreaterThanOrEqual),
            Token::And => Ok(BinaryOperator::And),
            Token::Or => Ok(BinaryOperator::Or),
            Token::Assign => Ok(BinaryOperator::Assign),
            Token::Pipeline => Ok(BinaryOperator::Pipeline),
            _ => Err(ParserError::UnexpectedToken(format!(
                "Unexpected token for binary operator: {:?}",
                token
            ))),
        }
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
            Some(Token::Assign) => Precedence::Assignment,
            Some(Token::Pipeline) => Precedence::Pipeline,
            Some(Token::Or) => Precedence::Or,
            Some(Token::And) => Precedence::And,
            Some(Token::Eq) | Some(Token::NotEq) => Precedence::Equality,
            Some(Token::Lt) | Some(Token::Gt) | Some(Token::LtEq) | Some(Token::GtEq) => {
                Precedence::Comparison
            }
            Some(Token::Plus) | Some(Token::Minus) => Precedence::Term,
            Some(Token::Multiply) | Some(Token::Divide) => Precedence::Factor,
            Some(Token::LParen) => Precedence::Call,
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
