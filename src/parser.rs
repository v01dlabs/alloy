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
        let token = self.tokens.pop_front();
        println!("advance: {:?}", token);
        token
    }

    /// Checks if the next token matches the expected token.
    fn check(&self, expected: &Token) -> bool {
        self.peek() == Some(expected)
    }

    /// Consumes the next token if it matches the expected token.
    fn match_token(&mut self, expected: &Token) -> bool {
        let current = self.peek();
        println!("match_token: expected {:?}, found {:?}", expected, current);
        if let Some(token) = current {
            if token == expected {
                self.advance();
                return true;
            }
        }
        false
    }

    /// Consumes the expected token or returns an error.
    fn consume(&mut self, expected: &Token) -> Result<(), String> {
        let current = self.peek();
        println!("consume: expected {:?}, found {:?}", expected, current);
        if let Some(token) = current {
            if token == expected {
                self.advance();
                return Ok(());
            }
        }
        Err(format!("Expected {:?}, found {:?}", expected, current))
    }

    /// Retrieves the recorded errors.
    pub fn get_errors(&self) -> &Vec<String> {
        &self.errors
    }

    /// Parses the entire program.
    pub fn parse(&mut self) -> Result<AstNode, String> {
        let mut declarations = Vec::new();
        while !self.is_at_end() {
            match self.peek() {
                Some(Token::Func) => declarations.push(self.parse_function_declaration()?),
                _ => declarations.push(self.parse_declaration()?),
            }
        }
        if declarations.is_empty() {
            Err("No valid declarations found".to_string())
        } else {
            Ok(AstNode::Program(declarations))
        }
    }

    /// Parses a primary expression.
    fn parse_primary(&mut self) -> Result<AstNode, String> {
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
            Some(Token::LBrace) => self.parse_trailing_closure(),
            Some(Token::LBracket) => self.parse_array_literal(),
            _ => Err("Unexpected token in primary expression".to_string()),
        }
    }

    /// Parses a declaration (function or variable).
    fn parse_declaration(&mut self) -> Result<AstNode, String> {
        match self.peek() {
            Some(Token::Let) => self.parse_variable_declaration(),
            Some(Token::Func) => self.parse_function_declaration(),
            _ => self.parse_statement(),
        }
    }

    /// Parses a function declaration.
    fn parse_function_declaration(&mut self) -> Result<AstNode, String> {
        self.consume(&Token::Func)?;
        let name = self.parse_identifier()?;

        let generic_params = if self.match_token(&Token::LBracket) {
            let params = self.parse_generic_params()?;
            self.consume(&Token::RBracket)?;
            Some(params)
        } else {
            None
        };

        self.consume(&Token::LParen)?;
        let params = self.parse_parameters()?;
        self.consume(&Token::RParen)?;

        let return_type = if self.match_token(&Token::Arrow) {
            Some(self.parse_type_annotation()?)
        } else {
            None
        };

        let body = self.parse_block()?;

        Ok(AstNode::FunctionDeclaration {
            name,
            generic_params,
            params,
            return_type,
            body,
        })
    }

    fn parse_function_call(&mut self, callee: AstNode) -> Result<AstNode, String> {
        let arguments = self.parse_arguments()?;

        if self.check(&Token::LBrace) {
            let closure = self.parse_trailing_closure()?;
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
    fn parse_trailing_closure(&mut self) -> Result<AstNode, String> {
        self.consume(&Token::LBrace)?;
        let body = self.parse_block()?;
        self.consume(&Token::RBrace)?;
        Ok(AstNode::Block(body))
    }

    // Helper method to parse generic parameters
    fn parse_generic_params(&mut self) -> Result<Vec<String>, String> {
        let mut params = Vec::new();
        while !self.check(&Token::RBracket) {
            params.push(self.parse_identifier()?);
            if !self.match_token(&Token::Comma) {
                break;
            }
        }
        Ok(params)
    }

    // Helper method to parse function parameters
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

    // Parses type annotations.
    fn parse_type_annotation(&mut self) -> Result<TypeAnnotation, String> {
        let base_type = self.parse_identifier()?;
        if self.match_token(&Token::LBracket) {
            let mut params = Vec::new();
            while !self.check(&Token::RBracket) {
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
        self.consume(&Token::Let)?;
        let name = self.parse_identifier()?;
        let type_annotation = if self.match_token(&Token::Colon) {
            Some(self.parse_type_annotation()?)
        } else {
            None
        };
        let initializer = if self.match_token(&Token::Assign) {
            Some(Box::new(self.parse_expression(Precedence::None)?))
        } else {
            None
        };
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
            Some(Token::LBrace) => Ok(AstNode::Block(self.parse_block()?)),
            _ => {
                let expr = self.parse_expression(Precedence::None)?;
                if self.match_token(&Token::Semicolon) {
                    Ok(expr)
                } else {
                    Err("Expected semicolon after expression statement".to_string())
                }
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
    fn parse_guard_statement(&mut self) -> Result<AstNode, String> {
        self.consume(&Token::Guard)?;
        let condition = Box::new(self.parse_expression(Precedence::None)?);
        self.consume(&Token::Else)?;
        let body = Box::new(self.parse_statement()?);
        Ok(AstNode::GuardStatement { condition, body })
    }

    /// Parses a return statement.
    fn parse_return_statement(&mut self) -> Result<AstNode, String> {
        self.advance(); // Consume 'return'
        let value = if !self.check(&Token::Semicolon) && !self.check(&Token::RBrace) {
            Some(Box::new(self.parse_expression(Precedence::None)?))
        } else {
            None
        };
        if self.check(&Token::Semicolon) {
            self.consume(&Token::Semicolon)?;
        }
        Ok(AstNode::ReturnStatement(value))
    }

    /// Parses a block of statements.
    fn parse_block(&mut self) -> Result<Vec<AstNode>, String> {
        self.consume(&Token::LBrace)?;
        let mut statements = Vec::new();
        while !self.check(&Token::RBrace) && !self.is_at_end() {
            statements.push(self.parse_declaration()?);
        }
        self.consume(&Token::RBrace)?;
        Ok(statements)
    }

    /// Parses an expression.
    fn parse_expression(&mut self, precedence: Precedence) -> Result<AstNode, String> {
        let mut left = self.parse_primary()?;

        while precedence < self.get_precedence() {
            if self.check(&Token::Semicolon) {
                break;
            }
            left = self.parse_infix(left)?;
        }

        Ok(left)
    }

    /// Parses an array literal.
    fn parse_array_literal(&mut self) -> Result<AstNode, String> {
        self.consume(&Token::LBracket)?;
        let mut elements = Vec::new();
        while !self.check(&Token::RBracket) {
            elements.push(self.parse_expression(Precedence::None)?);
            if !self.match_token(&Token::Comma) {
                break;
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
            Some(Token::Pipeline) => self.parse_pipeline(left),
            Some(Token::LParen) => self.parse_function_call(left),
            _ => Ok(left),
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

    /// Parses a pipeline operation.
    fn parse_pipeline(&mut self, left: AstNode) -> Result<AstNode, String> {
        self.advance(); // Consume the '|>' token
        let right = self.parse_expression(Precedence::Pipeline)?;
        Ok(AstNode::PipelineOperation {
            left: Box::new(left),
            right: Box::new(right),
        })
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

    /// Parses function arguments.
    fn parse_arguments(&mut self) -> Result<Vec<AstNode>, String> {
        self.consume(&Token::LParen)?;
        let mut arguments = Vec::new();
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
                Token::Let | Token::Func | Token::If | Token::While | Token::For | Token::Return
            ) {
                break;
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
    use crate::lexer::Lexer;
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
            Token::LBracket,
            Token::Identifier("T".to_string()),
            Token::RBracket,
            Token::LParen,
            Token::Identifier("a".to_string()),
            Token::Colon,
            Token::Identifier("T".to_string()),
            Token::Comma,
            Token::Identifier("b".to_string()),
            Token::Colon,
            Token::Identifier("T".to_string()),
            Token::RParen,
            Token::Arrow,
            Token::Identifier("T".to_string()),
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

        match result {
            AstNode::FunctionDeclaration {
                name,
                generic_params,
                params,
                return_type,
                body,
            } => {
                assert_eq!(name, "add");
                assert_eq!(generic_params, Some(vec!["T".to_string()]));
                assert_eq!(params.len(), 2);
                assert_eq!(params[0].0, "a");
                assert_eq!(params[1].0, "b");
                assert!(matches!(&params[0].1, TypeAnnotation::Simple(t) if t == "T"));
                assert!(matches!(&params[1].1, TypeAnnotation::Simple(t) if t == "T"));
                assert!(matches!(return_type, Some(TypeAnnotation::Simple(t)) if t == "T"));
                assert_eq!(body.len(), 1);
                match &body[0] {
                    AstNode::ReturnStatement(Some(box AstNode::BinaryOperation { .. })) => {}
                    _ => panic!("Expected return statement with binary operation"),
                }
            }
            _ => panic!("Expected FunctionDeclaration"),
        }
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
    fn test_parse_for_statement() {
        let source = "for name in names { print(name); }";
        let tokens = Lexer::<'_>::tokenize(source).unwrap();
        let mut parser = Parser::new(tokens);
        let result = parser.parse_statement();
        assert!(result.is_ok());
        if let Ok(AstNode::ForInLoop {
            item,
            iterable,
            body,
        }) = result
        {
            assert_eq!(item, "name");
            assert!(matches!(*iterable, AstNode::Identifier(ref i) if i == "names"));
            assert!(matches!(*body, AstNode::Block(_)));
        } else {
            panic!("Expected ForInLoop, got {:?}", result);
        }
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
        ];
        let mut parser = create_parser(tokens);
        let result = parser.parse().unwrap();
        if let AstNode::Program(declarations) = &result {
            assert_eq!(declarations.len(), 1);
            if let AstNode::VariableDeclaration {
                name,
                type_annotation,
                initializer,
            } = &declarations[0]
            {
                assert_eq!(name, "x");
                if let Some(TypeAnnotation::Generic(base_type, params)) = type_annotation {
                    assert_eq!(base_type, "Array");
                    assert_eq!(params.len(), 1);
                } else {
                    panic!("Expected Generic type annotation");
                }
                assert!(matches!(
                    initializer,
                    Some(box AstNode::ArrayLiteral { .. })
                ));
            } else {
                panic!("Expected VariableDeclaration");
            }
        } else {
            panic!("Expected Program");
        }
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
                generic_params: None,
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
        ];
        let mut parser = Parser::new(tokens);
        let result = parser.parse();
        assert!(result.is_ok());
        if let Ok(AstNode::Program(declarations)) = result {
            assert_eq!(declarations.len(), 2);
            match &declarations[0] {
                AstNode::VariableDeclaration { name, .. } => assert_eq!(name, "x"),
                _ => panic!("Expected VariableDeclaration for x"),
            }
            match &declarations[1] {
                AstNode::VariableDeclaration { name, .. } => assert_eq!(name, "y"),
                _ => panic!("Expected VariableDeclaration for y"),
            }
        } else {
            panic!("Expected a Program with two declarations");
        }
        assert_eq!(parser.get_errors().len(), 0);
    }

    #[test]
    fn test_parse_complex_program() {
        let source = r#"
        func processData[T](data: Array[T], predicate: func(T) -> bool) -> int {
            let mut sum = 0
            for value in data {
                if predicate(value) {
                    sum = sum + 1
                }
            }
            return sum
        }

        func main() {
            let numbers = [1, 2, 3, 4, 5]
            let result = processData[int](numbers) { value in
                value % 2 == 0
            }
            print("Even numbers count: \(result)")
        }
        "#;
        let tokens = Lexer::<'_>::tokenize(source).unwrap();
        let mut parser = Parser::new(tokens);
        let result = parser.parse();

        assert!(result.is_ok());
        if let Ok(AstNode::Program(declarations)) = result {
            assert_eq!(declarations.len(), 2);

            // Check processData function
            if let AstNode::FunctionDeclaration {
                name,
                generic_params,
                params,
                return_type,
                body,
            } = &declarations[0]
            {
                assert_eq!(name, "processData");
                assert_eq!(generic_params, &Some(vec!["T".to_string()]));
                assert_eq!(params.len(), 2);
                assert!(matches!(return_type, Some(TypeAnnotation::Simple(t)) if t == "int"));
                assert!(!body.is_empty());
            } else {
                panic!("Expected FunctionDeclaration for processData");
            }

            // Check main function
            if let AstNode::FunctionDeclaration {
                name,
                generic_params,
                params,
                return_type,
                body,
            } = &declarations[1]
            {
                assert_eq!(name, "main");
                assert_eq!(generic_params, &None);
                assert!(params.is_empty());
                assert_eq!(return_type, &None);
                assert!(!body.is_empty());
            } else {
                panic!("Expected FunctionDeclaration for main");
            }
        } else {
            panic!("Expected Program AST node");
        }
    }
}
