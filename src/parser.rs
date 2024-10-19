//! Parser for Alloy
//!
//! This module is responsible for taking a stream of tokens from the lexer
//! and constructing an Abstract Syntax Tree (AST) that represents the structure
//! of an Alloy program.

use crate::error::ParserError;
use crate::lexer::Token;
use std::iter::Peekable;
use std::vec::IntoIter;

/// Represents the precedence levels for operators.
#[derive(Debug, PartialEq, Eq, PartialOrd)]
pub enum Precedence {
    None,
    Assignment, // =
    Pipeline,   // |>
    Or,         // ||
    And,        // &&
    Equality,   // == !=
    Comparison, // < > <= >=
    Term,       // + -
    Factor,     // * /
    Unary,      // ! -
    Call,       // . ()
    Primary,
}

impl Precedence {
    fn from_token(token: &Token) -> Precedence {
        match token {
            Token::Eq | Token::NotEq => Precedence::Equality,
            Token::Lt | Token::LtEq | Token::Gt | Token::GtEq => Precedence::Comparison,
            Token::Plus | Token::Minus => Precedence::Term,
            Token::Multiply | Token::Divide => Precedence::Factor,
            Token::Not => Precedence::Unary,
            Token::And => Precedence::And,
            Token::Or => Precedence::Or,
            Token::Assign => Precedence::Assignment,
            Token::Pipeline => Precedence::Pipeline,
            Token::LParen => Precedence::Call,
            _ => Precedence::None,
        }
    }
}

/// Represents a node in the Abstract Syntax Tree (AST).
#[derive(Debug, Clone)]
pub enum AstNode {
    Program(Vec<AstNode>),
    FunctionDeclaration {
        name: String,
        generic_params: Option<Vec<String>>,
        params: Vec<(String, TypeAnnotation)>,
        return_type: Option<TypeAnnotation>,
        body: Vec<AstNode>,
    },
    VariableDeclaration {
        name: String,
        mutable: bool,
        type_annotation: Option<TypeAnnotation>,
        initializer: Option<Box<AstNode>>,
    },
    IfStatement {
        condition: Box<AstNode>,
        then_branch: Box<AstNode>,
        else_branch: Option<Box<AstNode>>,
    },
    WhileLoop {
        condition: Box<AstNode>,
        body: Box<AstNode>,
    },
    ForInLoop {
        item: String,
        iterable: Box<AstNode>,
        body: Box<AstNode>,
    },
    GuardStatement {
        condition: Box<AstNode>,
        body: Box<AstNode>,
    },
    ReturnStatement(Option<Box<AstNode>>),
    Block(Vec<AstNode>),
    BinaryOperation {
        left: Box<AstNode>,
        operator: BinaryOperator,
        right: Box<AstNode>,
    },
    UnaryOperation {
        operator: UnaryOperator,
        operand: Box<AstNode>,
    },
    FunctionCall {
        callee: Box<AstNode>,
        arguments: Vec<AstNode>,
    },
    GenericFunctionCall {
        name: String,
        generic_args: Vec<TypeAnnotation>,
        arguments: Vec<AstNode>,
    },
    TrailingClosure {
        callee: Box<AstNode>,
        closure: Box<AstNode>,
    },
    PipelineOperation {
        left: Box<AstNode>,
        right: Box<AstNode>,
    },
    Identifier(String),
    IntLiteral(i64),
    FloatLiteral(f64),
    StringLiteral(String),
    BoolLiteral(bool),
    ArrayLiteral(Vec<AstNode>),
}

/// Represents the type annotations in Alloy.
#[derive(Debug, Clone, PartialEq)]
pub enum TypeAnnotation {
    Simple(String),
    Generic(String, Vec<TypeAnnotation>),
    Function(Vec<TypeAnnotation>, Option<Box<TypeAnnotation>>),
    Int,
    Float,
    String,
    Bool,
    Array(Box<TypeAnnotation>),
    Custom(String),
}

/// Represents binary operators in Alloy.
#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    And,
    Or,
    Assign,
    Pipeline,
}

/// Represents unary operators in Alloy.
#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOperator {
    Negate,
    Not,
    Increment,
}

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
    pub fn parse(&mut self) -> Result<AstNode, ParserError> {
        let mut declarations = Vec::new();
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
            Ok(AstNode::Program(declarations))
        }
    }

    /// Parses a primary expression.
    fn parse_primary(&mut self) -> Result<AstNode, ParserError> {
        match self.advance() {
            Some(Token::Identifier(name)) => {
                if self.check(&Token::LParen) {
                    self.parse_function_call(AstNode::Identifier(name))
                } else {
                    Ok(AstNode::Identifier(name))
                }
            }
            Some(Token::IntLiteral(value)) => Ok(AstNode::IntLiteral(value)),
            Some(Token::FloatLiteral(value)) => Ok(AstNode::FloatLiteral(value)),
            Some(Token::StringLiteral(value)) => Ok(AstNode::StringLiteral(value)),
            Some(Token::BoolLiteral(value)) => Ok(AstNode::BoolLiteral(value)),
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
    pub fn parse_declaration(&mut self) -> Result<AstNode, ParserError> {
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
    fn parse_function_declaration(&mut self) -> Result<AstNode, ParserError> {
        self.consume(&Token::Fn)?;
        let name = self.parse_identifier()?;

        let generic_params = if self.consume_if(&Token::LBracket) {
            let params = self.parse_generic_params()?;
            self.consume(&Token::RBracket)?;
            Some(params)
        } else {
            None
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

        Ok(AstNode::FunctionDeclaration {
            name,
            generic_params,
            params,
            return_type,
            body,
        })
    }

    fn parse_function_call(&mut self, callee: AstNode) -> Result<AstNode, ParserError> {
        let arguments = self.parse_arguments()?;
        if self.check(&Token::LBrace) {
            let closure = self.finish_trailing_closure()?;
            Ok(AstNode::TrailingClosure {
                callee: Box::new(AstNode::FunctionCall {
                    callee: Box::new(callee),
                    arguments,
                }),
                closure: Box::new(closure),
            })
        } else {
            Ok(AstNode::FunctionCall {
                callee: Box::new(callee),
                arguments,
            })
        }
    }

    /// Finishes parsing a trailing closure.
    fn finish_trailing_closure(&mut self) -> Result<AstNode, ParserError> {
        let body = self.parse_block()?;
        Ok(AstNode::Block(body))
    }

    // Helper method to parse generic parameters
    fn parse_generic_params(&mut self) -> Result<Vec<String>, ParserError> {
        let mut params = Vec::new();
        while !self.check(&Token::RBracket) {
            params.push(self.parse_identifier()?);
            if !self.consume_if(&Token::Comma) {
                break;
            }
        }
        Ok(params)
    }

    // Helper method to parse function parameters
    fn parse_parameters(&mut self) -> Result<Vec<(String, TypeAnnotation)>, ParserError> {
        let mut params = Vec::new();
        if !self.check(&Token::RParen) {
            loop {
                let name = self.parse_identifier()?;
                self.consume(&Token::Colon)?;
                let type_annotation = self.parse_type_annotation()?;
                params.push((name, type_annotation));
                if !self.consume_if(&Token::Comma) {
                    break;
                }
            }
        }
        Ok(params)
    }

    /// Parses type annotations.
    fn parse_type_annotation(&mut self) -> Result<TypeAnnotation, ParserError> {
        if self.consume_if(&Token::Pipe) {
            let mut params = Vec::new();
            loop {
                if self.check(&Token::Pipe) {
                    break;
                }
                params.push(self.parse_type_annotation()?);
                if !self.consume_if(&Token::Comma) {
                    break;
                }
            }
            self.consume(&Token::Pipe)?;
            let return_type = if self.consume_if(&Token::Arrow) {
                Some(Box::new(self.parse_type_annotation()?))
            } else {
                None
            };
            return Ok(TypeAnnotation::Function(params, return_type));
        }
        let base_type = self.parse_identifier()?;
        if self.consume_if(&Token::LBracket) {
            let mut params = Vec::new();
            loop {
                params.push(self.parse_type_annotation()?);
                if !self.consume_if(&Token::Comma) {
                    break;
                }
            }
            self.consume(&Token::RBracket)?;
            Ok(TypeAnnotation::Generic(base_type, params))
        } else {
            Ok(TypeAnnotation::Simple(base_type))
        }
    }

    /// Parses a variable declaration.
    fn parse_variable_declaration(&mut self) -> Result<AstNode, ParserError> {
        self.consume(&Token::Let)?;
        let mutable = self.consume_if(&Token::Mut);
        let name = self.parse_identifier()?;
        let type_annotation = if self.consume_if(&Token::Colon) {
            Some(self.parse_type_annotation()?)
        } else {
            None
        };
        let initializer = if self.consume_if(&Token::Assign) {
            Some(Box::new(self.parse_expression(Precedence::None)?))
        } else {
            None
        };
        // Consume the semicolon if present, but don't require it
        self.consume_if(&Token::Semicolon);
        Ok(AstNode::VariableDeclaration {
            name,
            mutable,
            type_annotation,
            initializer,
        })
    }

    /// Parses a statement.
    pub fn parse_statement(&mut self) -> Result<AstNode, ParserError> {
        // Skip any leading newlines
        while self.consume_if(&Token::Newline) {}
        match self.peek() {
            Some(Token::If) => self.parse_if_statement(),
            Some(Token::While) => self.parse_while_statement(),
            Some(Token::For) => self.parse_for_statement(),
            Some(Token::Guard) => self.parse_guard_statement(),
            Some(Token::Return) => self.parse_return_statement(),
            Some(Token::LBrace) => Ok(AstNode::Block(self.parse_block()?)),
            Some(Token::Let) => self.parse_variable_declaration(),
            _ => self.parse_expression(Precedence::None).and_then(|expr| {
                self.consume_if(&Token::Semicolon);
                Ok(expr)
            }),
        }
    }

    /// Parses an if statement.
    fn parse_if_statement(&mut self) -> Result<AstNode, ParserError> {
        self.advance(); // Consume 'if'
        let paren = self.consume_if(&Token::LParen);
        let condition = self.parse_expression(Precedence::None)?;
        if paren {
            self.consume(&Token::RParen)?;
        }
        let then_branch = self.parse_statement_or_block()?;
        let else_branch = if self.consume_if(&Token::Else) {
            Some(Box::new(self.parse_statement_or_block()?))
        } else {
            None
        };
        Ok(AstNode::IfStatement {
            condition: Box::new(condition),
            then_branch: Box::new(then_branch),
            else_branch,
        })
    }

    /// Parses a while statement.
    fn parse_while_statement(&mut self) -> Result<AstNode, ParserError> {
        self.advance(); // Consume 'while'
        self.consume(&Token::LParen)?;
        let condition = self.parse_expression(Precedence::None)?;
        self.consume(&Token::RParen)?;
        let body = self.parse_block()?;
        Ok(AstNode::WhileLoop {
            condition: Box::new(condition),
            body: Box::new(AstNode::Block(body)),
        })
    }

    /// Parses a for statement.
    fn parse_for_statement(&mut self) -> Result<AstNode, ParserError> {
        self.consume(&Token::For)?;
        let item = self.parse_identifier()?;
        self.consume(&Token::In)?;
        let iterable = self.parse_expression(Precedence::None)?;
        let body = self.parse_block()?;
        Ok(AstNode::ForInLoop {
            item,
            iterable: Box::new(iterable),
            body: Box::new(AstNode::Block(body)),
        })
    }

    /// Parses a guard statement.
    fn parse_guard_statement(&mut self) -> Result<AstNode, ParserError> {
        self.consume(&Token::Guard)?;
        let condition = Box::new(self.parse_expression(Precedence::None)?);
        self.consume(&Token::Else)?;
        let body = Box::new(self.parse_statement()?);
        Ok(AstNode::GuardStatement { condition, body })
    }

    /// Parses a return statement.
    fn parse_return_statement(&mut self) -> Result<AstNode, ParserError> {
        self.advance(); // Consume 'return'
        let value = if !self.check(&Token::Semicolon) && !self.check(&Token::RBrace) {
            Some(Box::new(self.parse_expression(Precedence::None)?))
        } else {
            None
        };
        self.consume_if(&Token::Semicolon);
        Ok(AstNode::ReturnStatement(value))
    }

    fn parse_statement_or_block(&mut self) -> Result<AstNode, ParserError> {
        if self.check(&Token::LBrace) {
            Ok(AstNode::Block(self.parse_block()?))
        } else {
            Ok(self.parse_statement()?)
        }
    }

    /// Parses a block of statements.
    fn parse_block(&mut self) -> Result<Vec<AstNode>, ParserError> {
        self.consume(&Token::LBrace)?;

        let mut statements = Vec::new();
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

    fn parse_trailing_closure(&mut self, callee: AstNode) -> Result<AstNode, ParserError> {
        self.consume(&Token::Pipe)?;
        let mut arguments = Vec::new();
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
                callee: Box::new(callee),
                arguments,
            }),
            closure: Box::new(closure),
        })
    }

    /// Parses an expression.
    pub fn parse_expression(&mut self, precedence: Precedence) -> Result<AstNode, ParserError> {
        let mut left = self.parse_primary()?;

        while precedence < self.get_precedence() {
            if self.check(&Token::Semicolon) || self.is_at_end() {
                break;
            }
            left = self.parse_infix(left)?;
        }

        // Check for trailing closure
        if self.check(&Token::Pipe) {
            left = self.parse_trailing_closure(left)?;
        }

        Ok(left)
    }

    /// Parses an array literal.
    fn parse_array_literal(&mut self) -> Result<AstNode, ParserError> {
        let mut elements = Vec::new();
        while !self.check(&Token::RBracket) {
            elements.push(self.parse_expression(Precedence::None)?);

            if !self.consume_if(&Token::Comma) {
                break;
            }
        }
        println!("{:?}", elements);
        self.consume(&Token::RBracket)?;
        Ok(AstNode::ArrayLiteral(elements))
    }

    /// Parses an infix expression.
    fn parse_infix(&mut self, left: AstNode) -> Result<AstNode, ParserError> {
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
    fn parse_assignment(&mut self, left: AstNode) -> Result<AstNode, ParserError> {
        self.advance(); // Consume the '=' token
        let value = self.parse_expression(Precedence::Assignment)?;
        Ok(AstNode::BinaryOperation {
            left: Box::new(left),
            operator: BinaryOperator::Assign,
            right: Box::new(value),
        })
    }

    /// Parses a pipeline operation.
    fn parse_pipeline(&mut self, left: AstNode) -> Result<AstNode, ParserError> {
        self.advance(); // Consume the '|>' token
        let right = self.parse_expression(Precedence::Pipeline)?;
        Ok(AstNode::PipelineOperation {
            left: Box::new(left),
            right: Box::new(right),
        })
    }

    /// Parses a binary operation.
    fn parse_binary(
        &mut self,
        left: AstNode,
        precedence: Precedence,
    ) -> Result<AstNode, ParserError> {
        let operator = self.advance().ok_or(ParserError::UnexpectedToken(
            "Expected binary operator".to_string(),
        ))?;
        let right = self.parse_expression(precedence)?;
        Ok(AstNode::BinaryOperation {
            left: Box::new(left),
            operator: self.token_to_binary_operator(operator)?,
            right: Box::new(right),
        })
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
    fn parse_arguments(&mut self) -> Result<Vec<AstNode>, ParserError> {
        self.consume(&Token::LParen)?;
        let mut arguments = Vec::new();
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
    fn parse_identifier(&mut self) -> Result<String, ParserError> {
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
pub fn parse(tokens: Vec<Token>) -> Result<AstNode, ParserError> {
    let mut parser = Parser::new(tokens);
    parser.parse()
}
