//! Parser for the Alloy
//!
//! This module is responsible for taking a stream of tokens from the lexer
//! and constructing an Abstract Syntax Tree (AST) that represents the structure
//! of an Alloy program.

use crate::lexer::Token;
use std::iter::Peekable;
use std::vec::IntoIter;

/// Represents a node in the Abstract Syntax Tree.
#[derive(Debug, PartialEq, Clone)]
pub enum AstNode {
    Program(Vec<AstNode>),
    FunctionDeclaration {
        name: String,
        params: Vec<(String, TypeAnnotation)>,
        return_type: Option<TypeAnnotation>,
        body: Box<AstNode>,
    },
    VariableDeclaration {
        name: String,
        is_mutable: bool,
        type_annotation: Option<TypeAnnotation>,
        initializer: Option<Box<AstNode>>,
    },
    ForLoop {
        initializer: Option<Box<AstNode>>,
        condition: Option<Box<AstNode>>,
        increment: Option<Box<AstNode>>,
        body: Box<AstNode>,
    },
    WhileLoop {
        condition: Box<AstNode>,
        body: Box<AstNode>,
    },
    Block(Vec<AstNode>),
    ReturnStatement(Option<Box<AstNode>>),
    IfStatement {
        condition: Box<AstNode>,
        then_branch: Box<AstNode>,
        else_branch: Option<Box<AstNode>>,
    },
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
        function: Box<AstNode>,
        arguments: Vec<AstNode>,
    },
    ArrayLiteral(Vec<AstNode>),
    Identifier(String),
    IntegerLiteral(i64),
    FloatLiteral(f64),
    StringLiteral(String),
    BooleanLiteral(bool),
}

/// Represents the type annotations in Alloy.
#[derive(Debug, PartialEq, Clone)]
pub enum TypeAnnotation {
    Int,
    Float,
    String,
    Bool,
    Array(Box<TypeAnnotation>),
    Custom(String),
}

/// Represents binary operators in Alloy.
#[derive(Debug, PartialEq, Clone)]
pub enum BinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Equals,
    NotEquals,
    LessThan,
    GreaterThan,
    LessThanOrEqual,
    GreaterThanOrEqual,
    And,
    Or,
}

/// Represents unary operators in Alloy.
#[derive(Debug, PartialEq, Clone)]
pub enum UnaryOperator {
    Negate,
    Not,
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

    /// Parses the entire program.
    pub fn parse_program(&mut self) -> Result<AstNode, String> {
        let mut statements = Vec::new();
        while self.tokens.peek().is_some() {
            statements.push(self.parse_statement()?);
        }
        Ok(AstNode::Program(statements))
    }

    /// Parses a single statement.
    fn parse_statement(&mut self) -> Result<AstNode, String> {
        match self.tokens.peek() {
            Some(Token::Let) => self.parse_variable_declaration(),
            Some(Token::Func) => self.parse_function_declaration(),
            Some(Token::Return) => self.parse_return_statement(),
            Some(Token::If) => self.parse_if_statement(),
            Some(Token::While) => self.parse_while_loop(),
            Some(Token::For) => self.parse_for_loop(),
            Some(Token::LBracket) => self.parse_array_literal(),
            _ => self.parse_expression_statement(),
        }
    }

    /// Parses a variable declaration.
    fn parse_variable_declaration(&mut self) -> Result<AstNode, String> {
        self.consume(Token::Let)?;
        let is_mutable = self.match_token(Token::Mut);
        let name = self.consume_identifier()?;

        let type_annotation = if self.match_token(Token::Colon) {
            Some(self.parse_type()?)
        } else {
            None
        };

        let initializer = if self.match_token(Token::Assign) {
            Some(Box::new(self.parse_expression()?))
        } else {
            None
        };

        self.consume(Token::Semicolon)?;

        Ok(AstNode::VariableDeclaration {
            name,
            is_mutable,
            type_annotation,
            initializer,
        })
    }

    /// Parses a function declaration.
    fn parse_function_declaration(&mut self) -> Result<AstNode, String> {
        self.consume(Token::Func)?;
        let name = self.consume_identifier()?;
        self.consume(Token::LParen)?;

        let params = self.parse_parameters()?;

        self.consume(Token::RParen)?;

        let return_type = if self.match_token(Token::Arrow) {
            Some(self.parse_type()?)
        } else {
            None
        };

        let body = Box::new(self.parse_block()?);

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
                let name = self.consume_identifier()?;
                self.consume(Token::Colon)?;
                let type_annotation = self.parse_type()?;
                params.push((name, type_annotation));
                if !self.match_token(Token::Comma) {
                    break;
                }
            }
        }
        Ok(params)
    }

    /// Parses a type annotation.
    fn parse_type(&mut self) -> Result<TypeAnnotation, String> {
        if self.match_token(Token::LBracket) {
            let inner_type = Box::new(self.parse_type()?);
            self.consume(Token::RBracket)?;
            Ok(TypeAnnotation::Array(inner_type))
        } else {
            match self.tokens.next() {
                Some(Token::Int) => Ok(TypeAnnotation::Int),
                Some(Token::Float) => Ok(TypeAnnotation::Float),
                Some(Token::String) => Ok(TypeAnnotation::String),
                Some(Token::Bool) => Ok(TypeAnnotation::Bool),
                Some(Token::Identifier(name)) => Ok(TypeAnnotation::Custom(name)),
                _ => Err("Expected type annotation".to_string()),
            }
        }
    }

    /// Parses a block of statements.
    fn parse_block(&mut self) -> Result<AstNode, String> {
        self.consume(Token::LBrace)?;
        let mut statements = Vec::new();
        while !self.check(&Token::RBrace) {
            statements.push(self.parse_statement()?);
        }
        self.consume(Token::RBrace)?;
        Ok(AstNode::Block(statements))
    }

    /// Parses a return statement.
    fn parse_return_statement(&mut self) -> Result<AstNode, String> {
        self.consume(Token::Return)?;
        let value = if !self.check(&Token::Semicolon) {
            Some(Box::new(self.parse_expression()?))
        } else {
            None
        };
        self.consume(Token::Semicolon)?;
        Ok(AstNode::ReturnStatement(value))
    }

    /// Parses an if statement.
    fn parse_if_statement(&mut self) -> Result<AstNode, String> {
        self.consume(Token::If)?;
        self.consume(Token::LParen)?;
        let condition = Box::new(self.parse_expression()?);
        self.consume(Token::RParen)?;
        let then_branch = Box::new(self.parse_block()?);
        let else_branch = if self.match_token(Token::Else) {
            Some(Box::new(if self.check(&Token::If) {
                self.parse_if_statement()?
            } else {
                self.parse_block()?
            }))
        } else {
            None
        };
        Ok(AstNode::IfStatement {
            condition,
            then_branch,
            else_branch,
        })
    }

    /// Parses a while loop.
    fn parse_while_loop(&mut self) -> Result<AstNode, String> {
        self.consume(Token::While)?;
        self.consume(Token::LParen)?;
        let condition = Box::new(self.parse_expression()?);
        self.consume(Token::RParen)?;
        let body = Box::new(self.parse_block()?);
        Ok(AstNode::WhileLoop { condition, body })
    }

    /// Parses a for loop.
    fn parse_for_loop(&mut self) -> Result<AstNode, String> {
        self.consume(Token::For)?;
        self.consume(Token::LParen)?;
        let initializer = if !self.check(&Token::Semicolon) {
            Some(Box::new(self.parse_statement()?))
        } else {
            self.consume(Token::Semicolon)?;
            None
        };
        let condition = if !self.check(&Token::Semicolon) {
            Some(Box::new(self.parse_expression()?))
        } else {
            None
        };
        self.consume(Token::Semicolon)?;
        let increment = if !self.check(&Token::RParen) {
            Some(Box::new(self.parse_expression()?))
        } else {
            None
        };
        self.consume(Token::RParen)?;
        let body = Box::new(self.parse_block()?);
        Ok(AstNode::ForLoop {
            initializer,
            condition,
            increment,
            body,
        })
    }

    /// Parses an array literal.
    fn parse_array_literal(&mut self) -> Result<AstNode, String> {
        let mut elements = Vec::new();
        if !self.check(&Token::RBracket) {
            loop {
                elements.push(self.parse_expression()?);
                if !self.match_token(Token::Comma) {
                    break;
                }
            }
        }
        self.consume(Token::RBracket)?;
        Ok(AstNode::ArrayLiteral(elements))
    }

    /// Parses an expression statement.
    fn parse_expression_statement(&mut self) -> Result<AstNode, String> {
        let expr = self.parse_expression()?;
        self.consume(Token::Semicolon)?;
        Ok(expr)
    }

    /// Parses an expression.
    fn parse_expression(&mut self) -> Result<AstNode, String> {
        self.parse_assignment()
    }

    /// Parses an assignment expression.
    fn parse_assignment(&mut self) -> Result<AstNode, String> {
        let expr = self.parse_or()?;
        if self.match_token(Token::Assign) {
            let value = self.parse_assignment()?;
            Ok(AstNode::BinaryOperation {
                left: Box::new(expr),
                operator: BinaryOperator::Equals,
                right: Box::new(value),
            })
        } else {
            Ok(expr)
        }
    }

    /// Parses a logical OR expression.
    fn parse_or(&mut self) -> Result<AstNode, String> {
        let mut expr = self.parse_and()?;
        while self.match_token(Token::Or) {
            let right = Box::new(self.parse_and()?);
            expr = AstNode::BinaryOperation {
                left: Box::new(expr),
                operator: BinaryOperator::Or,
                right,
            };
        }
        Ok(expr)
    }

    /// Parses a logical AND expression.
    fn parse_and(&mut self) -> Result<AstNode, String> {
        let mut expr = self.parse_equality()?;
        while self.match_token(Token::And) {
            let right = Box::new(self.parse_equality()?);
            expr = AstNode::BinaryOperation {
                left: Box::new(expr),
                operator: BinaryOperator::And,
                right,
            };
        }
        Ok(expr)
    }

    /// Parses an equality expression.
    fn parse_equality(&mut self) -> Result<AstNode, String> {
        let mut expr = self.parse_comparison()?;
        while self.match_any(&[Token::Eq, Token::NotEq]) {
            let operator = match self.previous() {
                Token::Eq => BinaryOperator::Equals,
                Token::NotEq => BinaryOperator::NotEquals,
                _ => return Err("Unexpected token in equality expression".to_string()),
            };
            let right = Box::new(self.parse_comparison()?);
            expr = AstNode::BinaryOperation {
                left: Box::new(expr),
                operator,
                right,
            };
        }
        Ok(expr)
    }

    /// Parses a comparison expression.
    fn parse_comparison(&mut self) -> Result<AstNode, String> {
        let mut expr = self.parse_term()?;
        while self.match_any(&[Token::Lt, Token::Gt, Token::LtEq, Token::GtEq]) {
            let operator = match self.previous() {
                Token::Lt => BinaryOperator::LessThan,
                Token::Gt => BinaryOperator::GreaterThan,
                Token::LtEq => BinaryOperator::LessThanOrEqual,
                Token::GtEq => BinaryOperator::GreaterThanOrEqual,
                _ => return Err("Unexpected token in comparison expression".to_string()),
            };
            let right = Box::new(self.parse_term()?);
            expr = AstNode::BinaryOperation {
                left: Box::new(expr),
                operator,
                right,
            };
        }
        Ok(expr)
    }

    /// Parses a term (addition and subtraction).
    fn parse_term(&mut self) -> Result<AstNode, String> {
        let mut expr = self.parse_factor()?;
        while self.match_any(&[Token::Plus, Token::Minus]) {
            let operator = match self.previous() {
                Token::Plus => BinaryOperator::Add,
                Token::Minus => BinaryOperator::Subtract,
                _ => return Err("Unexpected token in term".to_string()),
            };
            let right = Box::new(self.parse_factor()?);
            expr = AstNode::BinaryOperation {
                left: Box::new(expr),
                operator,
                right,
            };
        }
        Ok(expr)
    }

    /// Parses a factor (multiplication and division).
    fn parse_factor(&mut self) -> Result<AstNode, String> {
        let mut expr = self.parse_unary()?;
        while self.match_any(&[Token::Multiply, Token::Divide]) {
            let operator = match self.previous() {
                Token::Multiply => BinaryOperator::Multiply,
                Token::Divide => BinaryOperator::Divide,
                _ => return Err("Unexpected token in factor".to_string()),
            };
            let right = Box::new(self.parse_unary()?);
            expr = AstNode::BinaryOperation {
                left: Box::new(expr),
                operator,
                right,
            };
        }
        Ok(expr)
    }

    /// Parses a unary expression.
    fn parse_unary(&mut self) -> Result<AstNode, String> {
        if self.match_any(&[Token::Minus, Token::Not]) {
            let operator = match self.previous() {
                Token::Minus => UnaryOperator::Negate,
                Token::Not => UnaryOperator::Not,
                _ => return Err("Unexpected unary operator".to_string()),
            };
            let operand = self.parse_unary()?;
            Ok(AstNode::UnaryOperation {
                operator,
                operand: Box::new(operand),
            })
        } else {
            self.parse_primary()
        }
    }

    /// Parses a function call.
    fn parse_call(&mut self) -> Result<AstNode, String> {
        let mut expr = self.parse_primary()?;
        loop {
            if self.match_token(Token::LParen) {
                expr = self.finish_call(expr)?;
            } else {
                break;
            }
        }
        Ok(expr)
    }

    /// Finishes parsing a function call.
    fn finish_call(&mut self, callee: AstNode) -> Result<AstNode, String> {
        let mut arguments = Vec::new();
        if !self.check(&Token::RParen) {
            loop {
                arguments.push(self.parse_expression()?);
                if !self.match_token(Token::Comma) {
                    break;
                }
            }
        }
        self.consume(Token::RParen)?;
        Ok(AstNode::FunctionCall {
            function: Box::new(callee),
            arguments,
        })
    }

    /// Parses a primary expression (literal or identifier).
    fn parse_primary(&mut self) -> Result<AstNode, String> {
        match self.tokens.next() {
            Some(Token::IntLiteral(value)) => Ok(AstNode::IntegerLiteral(value)),
            Some(Token::FloatLiteral(value)) => Ok(AstNode::FloatLiteral(value)),
            Some(Token::StringLiteral(value)) => Ok(AstNode::StringLiteral(value)),
            Some(Token::BoolLiteral(value)) => Ok(AstNode::BooleanLiteral(value)),
            Some(Token::Identifier(name)) => Ok(AstNode::Identifier(name)),
            Some(Token::LParen) => {
                let expr = self.parse_expression()?;
                self.consume(Token::RParen)?;
                Ok(expr)
            }
            Some(Token::LBracket) => self.parse_array_literal(),
            Some(token) => Err(format!(
                "Unexpected token in primary expression: {:?}",
                token
            )),
            None => Err("Unexpected end of input".to_string()),
        }
    }

    /// Consumes the expected token or returns an error.
    fn consume(&mut self, expected: Token) -> Result<(), String> {
        if let Some(token) = self.tokens.next() {
            if token == expected {
                Ok(())
            } else {
                Err(format!("Expected {:?}, found {:?}", expected, token))
            }
        } else {
            Err(format!("Expected {:?}, found end of input", expected))
        }
    }

    /// Consumes an identifier token and returns its value.
    fn consume_identifier(&mut self) -> Result<String, String> {
        match self.tokens.next() {
            Some(Token::Identifier(name)) => Ok(name),
            Some(token) => Err(format!("Expected identifier, found {:?}", token)),
            None => Err("Expected identifier, found end of input".to_string()),
        }
    }

    /// Checks if the next token matches the expected token without consuming it.
    fn check(&mut self, expected: &Token) -> bool {
        self.tokens.peek() == Some(expected)
    }

    /// Consumes the next token if it matches any of the given tokens.
    fn match_any(&mut self, tokens: &[Token]) -> bool {
        for token in tokens {
            if self.check(token) {
                self.tokens.next();
                return true;
            }
        }
        false
    }

    /// Consumes the next token if it matches the expected token.
    fn match_token(&mut self, expected: Token) -> bool {
        if self.check(&expected) {
            self.tokens.next();
            true
        } else {
            false
        }
    }

    /// Returns the previously consumed token.
    fn previous(&mut self) -> &Token {
        self.tokens.peek().expect("No previous token")
    }
}

/// Parses a vector of tokens into an AST.
pub fn parse(tokens: Vec<Token>) -> Result<AstNode, String> {
    let mut parser = Parser::new(tokens);
    parser.parse_program()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer;

    fn parse_code(input: &str) -> Result<AstNode, String> {
        let tokens = lexer::tokenize(input)?;
        parse(tokens)
    }

    #[test]
    fn test_variable_declaration() {
        let input = "let x: int = 5;";
        let ast = parse_code(input).unwrap();
        assert!(matches!(ast, AstNode::Program(ref statements) if statements.len() == 1));
        if let AstNode::Program(ref statements) = ast {
            assert!(matches!(&statements[0],
                AstNode::VariableDeclaration {
                    name,
                    is_mutable,
                    type_annotation,
                    initializer: Some(box AstNode::IntegerLiteral(5))
                } if name == "x" && !is_mutable &&
                    matches!(type_annotation, Some(TypeAnnotation::Int))
            ));
        }
    }

    #[test]
    fn test_function_declaration() {
        let input = "func add(a: int, b: int) -> int { return a + b; }";
        let ast = parse_code(input).unwrap();
        assert!(matches!(ast, AstNode::Program(ref statements) if statements.len() == 1));
        if let AstNode::Program(ref statements) = ast {
            assert!(matches!(&statements[0],
                AstNode::FunctionDeclaration {
                    name,
                    params,
                    return_type,
                    body: box AstNode::Block(_)
                } if name == "add" &&
                    params.len() == 2 &&
                    matches!(return_type, Some(TypeAnnotation::Int))
            ));
        }
    }

    #[test]
    fn test_if_statement() {
        let input = "if (x > 5) { return true; } else { return false; }";
        let ast = parse_code(input).unwrap();
        assert!(matches!(ast, AstNode::Program(ref statements) if statements.len() == 1));
        if let AstNode::Program(ref statements) = ast {
            assert!(matches!(&statements[0],
                AstNode::IfStatement {
                    condition: box AstNode::BinaryOperation { .. },
                    then_branch: box AstNode::Block(_),
                    else_branch: Some(box AstNode::Block(_))
                }
            ));
        }
    }

    #[test]
    fn test_while_loop() {
        let input = "while (i < 10) { i = i + 1; }";
        let ast = parse_code(input).unwrap();
        assert!(matches!(ast, AstNode::Program(ref statements) if statements.len() == 1));
        if let AstNode::Program(ref statements) = ast {
            assert!(matches!(&statements[0],
                AstNode::WhileLoop {
                    condition: box AstNode::BinaryOperation { .. },
                    body: box AstNode::Block(_)
                }
            ));
        }
    }

    #[test]
    fn test_for_loop() {
        let input = "for (let i = 0; i < 10; i = i + 1) { print(i); }";
        let ast = parse_code(input).unwrap();
        assert!(matches!(ast, AstNode::Program(ref statements) if statements.len() == 1));
        if let AstNode::Program(ref statements) = ast {
            assert!(matches!(&statements[0],
                AstNode::ForLoop {
                    initializer: Some(_),
                    condition: Some(_),
                    increment: Some(_),
                    body: box AstNode::Block(_)
                }
            ));
        }
    }

    #[test]
    fn test_array_literal() {
        let input = "let arr = [1, 2, 3, 4, 5];";
        let ast = parse_code(input).unwrap();
        assert!(matches!(ast, AstNode::Program(ref statements) if statements.len() == 1));
        if let AstNode::Program(ref statements) = ast {
            assert!(matches!(&statements[0],
                AstNode::VariableDeclaration {
                    initializer: Some(box AstNode::ArrayLiteral(elements)),
                    ..
                } if elements.len() == 5
            ));
        }
    }

    #[test]
    fn test_complex_expression() {
        let input = "let result = (a + b) * (c - d) / 2;";
        let ast = parse_code(input).unwrap();
        assert!(matches!(ast, AstNode::Program(ref statements) if statements.len() == 1));
        if let AstNode::Program(ref statements) = ast {
            assert!(matches!(&statements[0],
                AstNode::VariableDeclaration {
                    initializer: Some(box AstNode::BinaryOperation { .. }),
                    ..
                }
            ));
        }
    }
}
