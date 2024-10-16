//! Parser for Alloy
//!
//! This module is responsible for taking a stream of tokens from the lexer
//! and constructing an Abstract Syntax Tree (AST) that represents the structure
//! of an Alloy program.

use crate::lexer::Token;
use std::collections::VecDeque;

/// Represents the precedence levels for operators.
#[derive(Debug, PartialEq, Eq, PartialOrd)]
enum Precedence {
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
        params: Vec<(String, TypeAnnotation)>,
        return_type: Option<TypeAnnotation>,
        body: Vec<AstNode>,
    },
    VariableDeclaration {
        name: String,
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
    ForLoop {
        initializer: Option<Box<AstNode>>,
        condition: Option<Box<AstNode>>,
        increment: Option<Box<AstNode>>,
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
}

/// The Parser struct holds the state during parsing.
pub struct Parser {
    tokens: VecDeque<Token>,
    errors: Vec<String>,
}

impl Parser {
    /// Creates a new Parser instance.
    pub fn new(tokens: Vec<Token>) -> Self {
        Parser {
            tokens: tokens.into(),
            errors: Vec::new(),
        }
    }

    /// Peeks at the next token without consuming it.
    fn peek(&self) -> Option<&Token> {
        self.tokens.front()
    }

    /// Advances to the next token, consuming the current one.
    fn advance(&mut self) -> Option<Token> {
        self.tokens.pop_front()
    }

    /// Checks if the next token matches the expected token.
    fn check(&self, expected: &Token) -> bool {
        self.peek() == Some(expected)
    }

    /// Consumes the next token if it matches the expected token.
    fn match_token(&mut self, expected: &Token) -> bool {
        if self.check(expected) {
            self.advance();
            true
        } else {
            false
        }
    }

    /// Consumes the expected token or returns an error.
    fn consume(&mut self, expected: &Token) -> Result<(), String> {
        if self.match_token(expected) {
            Ok(())
        } else {
            Err(format!("Expected {:?}, found {:?}", expected, self.peek()))
        }
    }

    /// Retrieves the recorded errors.
    pub fn get_errors(&self) -> &Vec<String> {
        &self.errors
    }

    /// Parses the entire program.
    pub fn parse(&mut self) -> Result<AstNode, String> {
        let mut statements = Vec::new();
        while !self.is_at_end() {
            match self.parse_declaration() {
                Ok(stmt) => statements.push(stmt),
                Err(e) => {
                    self.errors.push(e.to_string());
                    self.synchronize();
                }
            }
        }
        Ok(AstNode::Program(statements))
    }

    /// Parses a declaration (function or variable).
    fn parse_declaration(&mut self) -> Result<AstNode, String> {
        let result = if self.match_token(&Token::Func) {
            self.parse_function_declaration()
        } else if self.match_token(&Token::Let) {
            self.parse_variable_declaration()
        } else {
            self.parse_statement()
        };

        if result.is_err() {
            self.synchronize();
        }

        result
    }

    /// Parses a function declaration.
    fn parse_function_declaration(&mut self) -> Result<AstNode, String> {
        let name = self.parse_identifier()?;
        self.consume(&Token::LParen)?;
        let params = self.parse_parameters()?;
        self.consume(&Token::RParen)?;

        let return_type = if self.match_token(&Token::Arrow) {
            Some(self.parse_type_annotation()?)
        } else {
            None
        };

        self.consume(&Token::LBrace)?;
        let body = self.parse_block()?;
        self.consume(&Token::RBrace)?;

        Ok(AstNode::FunctionDeclaration {
            name,
            params,
            return_type,
            body,
        })
    }

    /// Parses function parameters.
    fn parse_parameters(&mut self) -> Result<Vec<(String, TypeAnnotation)>, String> {
        let mut params = Vec::new();
        if !self.check(&Token::RParen) {
            loop {
                let name = self.parse_identifier()?;
                self.consume(&Token::Colon)?;
                let type_annotation = self.parse_type_annotation()?;
                params.push((name, type_annotation));
                if !self.match_token(&Token::Comma) {
                    break;
                }
            }
        }
        Ok(params)
    }

    /// Parses a type annotation.
    fn parse_type_annotation(&mut self) -> Result<TypeAnnotation, String> {
        let base_type = self.parse_identifier()?;
        if self.match_token(&Token::LBracket) {
            let mut params = Vec::new();
            loop {
                params.push(self.parse_type_annotation()?);
                if !self.match_token(&Token::Comma) {
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
    fn parse_variable_declaration(&mut self) -> Result<AstNode, String> {
        let name = self.parse_identifier()?;
        let type_annotation = if self.match_token(&Token::Colon) {
            let type_annotation = self.parse_type_annotation()?;
            Some(type_annotation)
        } else {
            None
        };
        let initializer = if self.match_token(&Token::Assign) {
            let initializer = self.parse_expression(Precedence::None)?;
            Some(Box::new(initializer))
        } else {
            None
        };
        if let Err(e) = self.consume(&Token::Semicolon) {
            self.errors.push(e.to_string());
            self.synchronize();
            return Err(e);
        }
        Ok(AstNode::VariableDeclaration {
            name,
            type_annotation,
            initializer,
        })
    }

    /// Parses a statement.
    fn parse_statement(&mut self) -> Result<AstNode, String> {
        match self.peek() {
            Some(Token::If) => self.parse_if_statement(),
            Some(Token::While) => self.parse_while_statement(),
            Some(Token::For) => self.parse_for_statement(),
            Some(Token::Guard) => self.parse_guard_statement(),
            Some(Token::Return) => self.parse_return_statement(),
            Some(Token::LBrace) => self.parse_block().map(AstNode::Block),
            _ => {
                let expr = self.parse_expression(Precedence::None)?;
                self.consume(&Token::Semicolon)?;
                Ok(expr)
            }
        }
    }

    /// Parses an if statement.
    fn parse_if_statement(&mut self) -> Result<AstNode, String> {
        self.advance(); // Consume 'if'
        self.consume(&Token::LParen)?;
        let condition = self.parse_expression(Precedence::None)?;
        self.consume(&Token::RParen)?;
        let then_branch = self.parse_statement()?;
        let else_branch = if self.match_token(&Token::Else) {
            Some(Box::new(self.parse_statement()?))
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
    fn parse_while_statement(&mut self) -> Result<AstNode, String> {
        self.advance(); // Consume 'while'
        self.consume(&Token::LParen)?;
        let condition = self.parse_expression(Precedence::None)?;
        self.consume(&Token::RParen)?;
        let body = Box::new(self.parse_statement()?);
        Ok(AstNode::WhileLoop {
            condition: Box::new(condition),
            body,
        })
    }

    /// Parses a for statement.
    fn parse_for_statement(&mut self) -> Result<AstNode, String> {
        self.advance(); // Consume 'for'
        self.consume(&Token::LParen)?;
        let initializer = if !self.check(&Token::Semicolon) {
            Some(Box::new(self.parse_variable_declaration()?))
        } else {
            self.advance(); // Consume semicolon
            None
        };
        let condition = if !self.check(&Token::Semicolon) {
            Some(Box::new(self.parse_expression(Precedence::None)?))
        } else {
            None
        };
        self.consume(&Token::Semicolon)?;
        let increment = if !self.check(&Token::RParen) {
            Some(Box::new(self.parse_expression(Precedence::None)?))
        } else {
            None
        };
        self.consume(&Token::RParen)?;
        let body = Box::new(self.parse_statement()?);
        Ok(AstNode::ForLoop {
            initializer,
            condition,
            increment,
            body,
        })
    }

    /// Parses a guard statement.
    fn parse_guard_statement(&mut self) -> Result<AstNode, String> {
        self.advance(); // Consume 'guard'
        let condition = self.parse_expression(Precedence::None)?;
        self.consume(&Token::Else)?;
        let body = self.parse_statement()?;
        Ok(AstNode::GuardStatement {
            condition: Box::new(condition),
            body: Box::new(body),
        })
    }

    /// Parses a return statement.
    fn parse_return_statement(&mut self) -> Result<AstNode, String> {
        self.advance(); // Consume 'return'
        let value = if !self.check(&Token::Semicolon) {
            Some(Box::new(self.parse_expression(Precedence::None)?))
        } else {
            None
        };
        self.consume(&Token::Semicolon)?;
        Ok(AstNode::ReturnStatement(value))
    }

    /// Parses a block of statements.
    fn parse_block(&mut self) -> Result<Vec<AstNode>, String> {
        let mut statements = Vec::new();
        while !self.check(&Token::RBrace) && !self.is_at_end() {
            statements.push(self.parse_declaration()?);
        }
        Ok(statements)
    }

    /// Parses an expression.
    fn parse_expression(&mut self, precedence: Precedence) -> Result<AstNode, String> {
        let mut expr = self.parse_prefix()?;

        while precedence < self.get_precedence() {
            expr = self.parse_infix(expr)?;
        }

        if self.check(&Token::LBrace) {
            expr = self.finish_trailing_closure(expr)?;
        }

        Ok(expr)
    }

    /// Parses a prefix expression.
    fn parse_prefix(&mut self) -> Result<AstNode, String> {
        match self.advance() {
            Some(Token::Identifier(name)) => {
                if self.check(&Token::LParen) {
                    self.parse_call(AstNode::Identifier(name))
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
            Some(Token::Minus) => {
                let right = self.parse_expression(Precedence::Unary)?;
                Ok(AstNode::UnaryOperation {
                    operator: UnaryOperator::Negate,
                    operand: Box::new(right),
                })
            }
            Some(Token::Not) => {
                let right = self.parse_expression(Precedence::Unary)?;
                Ok(AstNode::UnaryOperation {
                    operator: UnaryOperator::Not,
                    operand: Box::new(right),
                })
            }
            Some(Token::LBracket) => self.parse_array_literal(),
            token => {
                Err(format!("Unexpected token in expression: {:?}", token))
            }
        }
    }

    /// Parses an array literal.
    fn parse_array_literal(&mut self) -> Result<AstNode, String> {
        let mut elements = Vec::new();

        while !self.check(&Token::RBracket) {
            elements.push(self.parse_expression(Precedence::None)?);
            if !self.check(&Token::RBracket) {
                self.consume(&Token::Comma)?;
            }
        }

        self.consume(&Token::RBracket)?;
        Ok(AstNode::ArrayLiteral(elements))
    }

    /// Parses an infix expression.
    fn parse_infix(&mut self, left: AstNode) -> Result<AstNode, String> {
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
            Some(Token::LParen) => self.parse_call(left),
            Some(Token::LBrace) => self.finish_trailing_closure(left),
            Some(Token::Pipeline) => {
                self.advance();
                let right = self.parse_expression(Precedence::Pipeline)?;
                Ok(AstNode::PipelineOperation {
                    left: Box::new(left),
                    right: Box::new(right),
                })
            }
            _ => Ok(left),
        }
    }

    /// Parses a binary operation.
    fn parse_binary(&mut self, left: AstNode, precedence: Precedence) -> Result<AstNode, String> {
        let operator = self.advance().ok_or("Expected binary operator")?;
        let right = self.parse_expression(precedence)?;
        Ok(AstNode::BinaryOperation {
            left: Box::new(left),
            operator: self.token_to_binary_operator(operator)?,
            right: Box::new(right),
        })
    }

    /// Converts a token to a binary operator.
    fn token_to_binary_operator(&self, token: Token) -> Result<BinaryOperator, String> {
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
            _ => Err(format!("Unexpected token for binary operator: {:?}", token)),
        }
    }

    /// Parses an assignment operation.
    fn parse_assignment(&mut self, left: AstNode) -> Result<AstNode, String> {
        self.advance(); // Consume the '=' token
        let value = self.parse_expression(Precedence::Assignment)?;
        Ok(AstNode::BinaryOperation {
            left: Box::new(left),
            operator: BinaryOperator::Assign,
            right: Box::new(value),
        })
    }

    /// Parses a function call.
    fn parse_call(&mut self, callee: AstNode) -> Result<AstNode, String> {
        let arguments = self.parse_arguments()?;
        Ok(AstNode::FunctionCall {
            callee: Box::new(callee),
            arguments,
        })
    }

    /// Parses function arguments.
    fn parse_arguments(&mut self) -> Result<Vec<AstNode>, String> {
        let mut arguments = Vec::new();
        self.consume(&Token::LParen)?;
        if !self.check(&Token::RParen) {
            loop {
                arguments.push(self.parse_expression(Precedence::None)?);
                if !self.match_token(&Token::Comma) {
                    break;
                }
            }
        }
        self.consume(&Token::RParen)?;
        Ok(arguments)
    }

    /// Parses a pipeline operation.
    fn parse_pipeline(&mut self, left: AstNode) -> Result<AstNode, String> {
        self.advance(); // Consume the '|>' token
        let right = self.parse_expression(Precedence::Call)?;
        Ok(AstNode::PipelineOperation {
            left: Box::new(left),
            right: Box::new(right),
        })
    }

    /// Finishes parsing a trailing closure.
    fn finish_trailing_closure(&mut self, callee: AstNode) -> Result<AstNode, String> {
        self.advance(); // Consume '{'
        let body = self.parse_block()?;
        self.consume(&Token::RBrace)?;
        Ok(AstNode::TrailingClosure {
            callee: Box::new(callee),
            closure: Box::new(AstNode::Block(body)),
        })
    }

    /// Parses a closure.
    fn parse_closure(&mut self) -> Result<AstNode, String> {
        self.consume(&Token::LBrace)?;
        let body = self.parse_block()?;
        self.consume(&Token::RBrace)?;
        Ok(AstNode::Block(body))
    }

    /// Gets the precedence of the current token.
    fn get_precedence(&self) -> Precedence {
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
    fn parse_identifier(&mut self) -> Result<String, String> {
        match self.advance() {
            Some(Token::Identifier(name)) => Ok(name),
            _ => Err("Expected identifier".to_string()),
        }
    }

    /// Synchronizes the parser after an error.
    fn synchronize(&mut self) {
        self.advance();
        while let Some(token) = self.peek() {
            if matches!(
                token,
                Token::Semicolon
                    | Token::Func
                    | Token::Let
                    | Token::For
                    | Token::If
                    | Token::While
                    | Token::Return
            ) {
                return;
            }
            self.advance();
        }
    }

    /// Checks if the parser has reached the end of the token stream.
    fn is_at_end(&self) -> bool {
        self.peek() == Some(&Token::Eof) || self.peek().is_none()
    }
}

/// Parses a vector of tokens into an AST.
pub fn parse(tokens: Vec<Token>) -> Result<AstNode, String> {
    let mut parser = Parser::new(tokens);
    parser.parse()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Token;

    // Helper function to create a parser from a vector of tokens
    fn create_parser(tokens: Vec<Token>) -> Parser {
        Parser::new(tokens)
    }

    #[test]
    fn test_parse_variable_declaration() {
        let tokens = vec![
            Token::Let,
            Token::Identifier("x".to_string()),
            Token::Colon,
            Token::Identifier("int".to_string()),
            Token::Assign,
            Token::IntLiteral(5),
            Token::Semicolon,
        ];
        let mut parser = create_parser(tokens);
        let result = parser.parse_declaration().unwrap();
        assert!(matches!(result,
            AstNode::VariableDeclaration {
                name,
                type_annotation: Some(TypeAnnotation::Simple(type_name)),
                initializer: Some(box AstNode::IntLiteral(5))
            } if name == "x" && type_name == "int"
        ));
    }

    #[test]
    fn test_parse_function_declaration() {
        let tokens = vec![
            Token::Func,
            Token::Identifier("add".to_string()),
            Token::LParen,
            Token::Identifier("a".to_string()),
            Token::Colon,
            Token::Identifier("int".to_string()),
            Token::Comma,
            Token::Identifier("b".to_string()),
            Token::Colon,
            Token::Identifier("int".to_string()),
            Token::RParen,
            Token::Arrow,
            Token::Identifier("int".to_string()),
            Token::LBrace,
            Token::Return,
            Token::Identifier("a".to_string()),
            Token::Plus,
            Token::Identifier("b".to_string()),
            Token::Semicolon,
            Token::RBrace,
        ];
        let mut parser = create_parser(tokens);
        let result = parser.parse_declaration().unwrap();
        assert!(matches!(result,
            AstNode::FunctionDeclaration {
                name,
                params,
                return_type: Some(TypeAnnotation::Simple(return_type)),
                body
            } if name == "add" && params.len() == 2 && return_type == "int" && body.len() == 1
        ));
    }

    #[test]
    fn test_parse_if_statement() {
        let tokens = vec![
            Token::If,
            Token::LParen,
            Token::Identifier("x".to_string()),
            Token::Gt,
            Token::IntLiteral(5),
            Token::RParen,
            Token::LBrace,
            Token::Return,
            Token::BoolLiteral(true),
            Token::Semicolon,
            Token::RBrace,
            Token::Else,
            Token::LBrace,
            Token::Return,
            Token::BoolLiteral(false),
            Token::Semicolon,
            Token::RBrace,
        ];
        let mut parser = create_parser(tokens);
        let result = parser.parse_statement().unwrap();
        assert!(matches!(result,
            AstNode::IfStatement {
                condition: box AstNode::BinaryOperation { .. },
                then_branch: box AstNode::Block(then_statements),
                else_branch: Some(box AstNode::Block(else_statements))
            } if then_statements.len() == 1 && else_statements.len() == 1
        ));
    }

    #[test]
    fn test_parse_while_loop() {
        let tokens = vec![
            Token::While,
            Token::LParen,
            Token::Identifier("i".to_string()),
            Token::Lt,
            Token::IntLiteral(10),
            Token::RParen,
            Token::LBrace,
            Token::Identifier("i".to_string()),
            Token::Assign,
            Token::Identifier("i".to_string()),
            Token::Plus,
            Token::IntLiteral(1),
            Token::Semicolon,
            Token::RBrace,
        ];
        let mut parser = create_parser(tokens);
        let result = parser.parse_statement().unwrap();
        assert!(matches!(result,
            AstNode::WhileLoop {
                condition: box AstNode::BinaryOperation { .. },
                body: box AstNode::Block(statements)
            } if statements.len() == 1
        ));
    }

    #[test]
    fn test_parse_for_loop() {
        let tokens = vec![
            Token::For,
            Token::LParen,
            Token::Let,
            Token::Identifier("i".to_string()),
            Token::Assign,
            Token::IntLiteral(0),
            Token::Semicolon,
            Token::Identifier("i".to_string()),
            Token::Lt,
            Token::IntLiteral(10),
            Token::Semicolon,
            Token::Identifier("i".to_string()),
            Token::Assign,
            Token::Identifier("i".to_string()),
            Token::Plus,
            Token::IntLiteral(1),
            Token::RParen,
            Token::LBrace,
            Token::RBrace,
        ];
        let mut parser = create_parser(tokens);
        let result = parser.parse_statement().unwrap();
        assert!(matches!(result,
            AstNode::ForLoop {
                initializer: Some(box AstNode::VariableDeclaration { .. }),
                condition: Some(box AstNode::BinaryOperation { .. }),
                increment: Some(box AstNode::BinaryOperation { .. }),
                body: box AstNode::Block(_)
            }
        ));
    }

    #[test]
    fn test_parse_pipeline_operator() {
        let tokens = vec![
            Token::Identifier("x".to_string()),
            Token::Pipeline,
            Token::Identifier("foo".to_string()),
            Token::LParen,
            Token::RParen,
            Token::Pipeline,
            Token::Identifier("bar".to_string()),
            Token::LParen,
            Token::RParen,
            Token::Semicolon,
        ];
        let mut parser = create_parser(tokens);
        let result = parser.parse_statement().unwrap();
        assert!(matches!(result,
            AstNode::PipelineOperation {
                left: box AstNode::PipelineOperation { .. },
                right: box AstNode::FunctionCall { .. }
            }
        ));
    }

    #[test]
    fn test_parse_trailing_closure() {
        let tokens = vec![
            Token::Identifier("someFunction".to_string()),
            Token::LParen,
            Token::RParen,
            Token::LBrace,
            Token::Return,
            Token::IntLiteral(42),
            Token::Semicolon,
            Token::RBrace,
        ];
        let mut parser = create_parser(tokens);
        let result = parser.parse_expression(Precedence::None).unwrap();
        assert!(matches!(result,
            AstNode::TrailingClosure {
                callee: box AstNode::FunctionCall { .. },
                closure: box AstNode::Block(statements)
            } if statements.len() == 1
        ));
    }

    #[test]
    fn test_parse_guard_statement() {
        let tokens = vec![
            Token::Guard,
            Token::Identifier("x".to_string()),
            Token::Gt,
            Token::IntLiteral(0),
            Token::Else,
            Token::LBrace,
            Token::Return,
            Token::Semicolon,
            Token::RBrace,
        ];
        let mut parser = create_parser(tokens);
        let result = parser.parse_statement().unwrap();
        assert!(matches!(result,
            AstNode::GuardStatement {
                condition: box AstNode::BinaryOperation { .. },
                body: box AstNode::Block(statements)
            } if statements.len() == 1
        ));
    }

    #[test]
    fn test_parse_return_statement() {
        let tokens = vec![
            Token::Return,
            Token::Identifier("x".to_string()),
            Token::Plus,
            Token::Identifier("y".to_string()),
            Token::Semicolon,
        ];
        let mut parser = create_parser(tokens);
        let result = parser.parse_statement().unwrap();
        assert!(matches!(
            result,
            AstNode::ReturnStatement(Some(box AstNode::BinaryOperation { .. }))
        ));
    }

    #[test]
    fn test_parse_complex_expression() {
        let tokens = vec![
            Token::Identifier("a".to_string()),
            Token::Plus,
            Token::Identifier("b".to_string()),
            Token::Multiply,
            Token::LParen,
            Token::Identifier("c".to_string()),
            Token::Minus,
            Token::Identifier("d".to_string()),
            Token::RParen,
            Token::Semicolon,
        ];
        let mut parser = create_parser(tokens);
        let result = parser.parse_statement().unwrap();
        assert!(matches!(result,
            AstNode::BinaryOperation {
                left: box AstNode::Identifier(a),
                operator: BinaryOperator::Add,
                right: box AstNode::BinaryOperation {
                    left: box AstNode::Identifier(b),
                    operator: BinaryOperator::Multiply,
                    right: box AstNode::BinaryOperation { .. }
                }
            } if a == "a" && b == "b"
        ));
    }

    #[test]
    fn test_parse_generic_type_annotation() {
        let tokens = vec![
            Token::Let,
            Token::Identifier("x".to_string()),
            Token::Colon,
            Token::Identifier("Array".to_string()),
            Token::LBracket,
            Token::Identifier("int".to_string()),
            Token::RBracket,
            Token::Assign,
            Token::LBracket,
            Token::IntLiteral(1),
            Token::Comma,
            Token::IntLiteral(2),
            Token::Comma,
            Token::IntLiteral(3),
            Token::RBracket,
            Token::Semicolon,
        ];
        let mut parser = create_parser(tokens);
        let result = parser.parse_declaration().unwrap();
        assert!(matches!(result,
            AstNode::VariableDeclaration {
                name,
                type_annotation: Some(TypeAnnotation::Generic(base_type, params)),
                initializer: Some(box AstNode::ArrayLiteral { .. })
            } if name == "x" && base_type == "Array" && params.len() == 1
        ));
    }

    #[test]
    fn test_parse_nested_generic_type_annotation() {
        let tokens = vec![
            Token::Let,
            Token::Identifier("x".to_string()),
            Token::Colon,
            Token::Identifier("Map".to_string()),
            Token::LBracket,
            Token::Identifier("String".to_string()),
            Token::Comma,
            Token::Identifier("Array".to_string()),
            Token::LBracket,
            Token::Identifier("int".to_string()),
            Token::RBracket,
            Token::RBracket,
            Token::Semicolon,
        ];
        let mut parser = create_parser(tokens);
        let result = parser.parse_declaration().unwrap();
        assert!(matches!(result,
            AstNode::VariableDeclaration {
                name,
                type_annotation: Some(TypeAnnotation::Generic(base_type, params)),
                initializer: None
            } if name == "x" && base_type == "Map" && params.len() == 2
        ));
    }

    #[test]
    fn test_parse_function_with_generic_return_type() {
        let tokens = vec![
            Token::Func,
            Token::Identifier("getValues".to_string()),
            Token::LParen,
            Token::RParen,
            Token::Arrow,
            Token::Identifier("Array".to_string()),
            Token::LBracket,
            Token::Identifier("int".to_string()),
            Token::RBracket,
            Token::LBrace,
            Token::Return,
            Token::LBracket,
            Token::IntLiteral(1),
            Token::Comma,
            Token::IntLiteral(2),
            Token::Comma,
            Token::IntLiteral(3),
            Token::RBracket,
            Token::Semicolon,
            Token::RBrace,
        ];
        let mut parser = create_parser(tokens);
        let result = parser.parse_declaration().unwrap();
        assert!(matches!(result,
            AstNode::FunctionDeclaration {
                name,
                params,
                return_type: Some(TypeAnnotation::Generic(base_type, type_params)),
                body
            } if name == "getValues" && params.is_empty() && base_type == "Array" && type_params.len() == 1 && body.len() == 1
        ));
    }

    #[test]
    fn test_parse_error_recovery() {
        let tokens = vec![
            Token::Let,
            Token::Identifier("x".to_string()),
            Token::Assign,
            Token::IntLiteral(5),
            Token::Let,
            Token::Identifier("y".to_string()),
            Token::Assign,
            Token::IntLiteral(10),
            Token::Semicolon,
        ];
        let mut parser = Parser::new(tokens);
        let _result = parser.parse_declaration(); // Prefix with underscore to suppress warning
        let errors = parser.get_errors();
        assert_eq!(errors.len(), 1); // Expecting 1 error due to missing semicolon
        assert_eq!(errors[0], "Expected Semicolon, found Some(Let)");
    }

    #[test]
    fn test_parse_complex_program() {
        let tokens = vec![
            Token::Func,
            Token::Identifier("processData".to_string()),
            Token::LParen,
            Token::Identifier("data".to_string()),
            Token::Colon,
            Token::Identifier("Array".to_string()),
            Token::LBracket,
            Token::Identifier("int".to_string()),
            Token::RBracket,
            Token::RParen,
            Token::Arrow,
            Token::Identifier("int".to_string()),
            Token::LBrace,
            Token::Let,
            Token::Identifier("sum".to_string()),
            Token::Assign,
            Token::IntLiteral(0),
            Token::Semicolon,
            Token::For,
            Token::LParen,
            Token::Let,
            Token::Identifier("i".to_string()),
            Token::Assign,
            Token::IntLiteral(0),
            Token::Semicolon,
            Token::Identifier("i".to_string()),
            Token::Lt,
            Token::Identifier("data".to_string()),
            Token::Dot,
            Token::Identifier("length".to_string()),
            Token::Semicolon,
            Token::Identifier("i".to_string()),
            Token::Assign,
            Token::Identifier("i".to_string()),
            Token::Plus,
            Token::IntLiteral(1),
            Token::RParen,
            Token::LBrace,
            Token::If,
            Token::LParen,
            Token::Identifier("data".to_string()),
            Token::LBracket,
            Token::Identifier("i".to_string()),
            Token::RBracket,
            Token::Gt,
            Token::IntLiteral(0),
            Token::RParen,
            Token::LBrace,
            Token::Identifier("sum".to_string()),
            Token::Assign,
            Token::Identifier("sum".to_string()),
            Token::Plus,
            Token::Identifier("data".to_string()),
            Token::LBracket,
            Token::Identifier("i".to_string()),
            Token::RBracket,
            Token::Semicolon,
            Token::RBrace,
            Token::RBrace,
            Token::Return,
            Token::Identifier("sum".to_string()),
            Token::Semicolon,
            Token::RBrace,
        ];
        let mut parser = create_parser(tokens);
        let result = parser.parse();
        assert!(result.is_ok());
        if let Ok(AstNode::Program(statements)) = result {
            assert_eq!(statements.len(), 1);
            assert!(matches!(statements[0], AstNode::FunctionDeclaration { .. }));
        } else {
            panic!("Expected Program node");
        }
    }
}
