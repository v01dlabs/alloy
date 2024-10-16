//! Lexical analysis for Alloy
//!
//! The lexer is responsible for converting raw source code into
//! a series of tokens that can be processed by the parser.

use std::str::Chars;
use std::iter::Peekable;

/// Represents a token in Alloy.
#[derive(Debug, PartialEq, Clone)]
pub enum Token {
    // Keywords
    Let,
    Mut,
    Func,
    Return,
    If,
    Else,
    While,
    For,
    In,
    Async,
    Await,
    Guard,

    // Types
    Int,
    Float,
    String,
    Bool,

    // Literals
    IntLiteral(i64),
    FloatLiteral(f64),
    StringLiteral(String),
    BoolLiteral(bool),

    // Identifiers
    Identifier(String),

    // Operators
    Plus,
    Minus,
    Multiply,
    Divide,
    Assign,
    Eq,
    NotEq,
    Lt,
    Gt,
    LtEq,
    GtEq,
    And,
    Or,
    Not,

    // Delimiters
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Comma,
    Colon,
    Semicolon,
    Arrow,

    // End of file
    Eof,
}

/// The Lexer struct handles the conversion of source code into tokens.
pub struct Lexer<'a> {
    input: Peekable<Chars<'a>>,
    line: usize,
    column: usize,
}

impl<'a> Lexer<'a> {
    /// Creates a new Lexer instance from the given input string.
    pub fn new(input: &'a str) -> Self {
        Lexer {
            input: input.chars().peekable(),
            line: 1,
            column: 1,
        }
    }

    /// Advances the lexer to the next character in the input.
    fn advance(&mut self) -> Option<char> {
        let ch = self.input.next();
        if let Some(c) = ch {
            if c == '\n' {
                self.line += 1;
                self.column = 1;
            } else {
                self.column += 1;
            }
        }
        ch
    }

    /// Peeks at the next character in the input without advancing the lexer.
    fn peek(&mut self) -> Option<&char> {
        self.input.peek()
    }

    /// Skips over any whitespace characters in the input.
    fn skip_whitespace(&mut self) {
        while let Some(&c) = self.peek() {
            if !c.is_whitespace() {
                break;
            }
            self.advance();
        }
    }

    /// Reads an identifier or keyword from the input.
    fn read_identifier(&mut self, first_char: char) -> String {
        let mut identifier = String::new();
        identifier.push(first_char);

        while let Some(&c) = self.peek() {
            if c.is_alphanumeric() || c == '_' {
                identifier.push(self.advance().unwrap());
            } else {
                break;
            }
        }

        identifier
    }

    /// Reads a number (integer or float) from the input.
    fn read_number(&mut self, first_char: char) -> Result<Token, String> {
        let mut number = String::new();
        number.push(first_char);
        let mut is_float = false;

        while let Some(&c) = self.peek() {
            if c.is_ascii_digit() || (c == '.' && !is_float) || (c.to_ascii_lowercase() == 'e' && !number.contains('e') && !number.contains('E')) {
                if c == '.' {
                    is_float = true;
                }
                number.push(self.advance().unwrap());
                if c.to_ascii_lowercase() == 'e' {
                    // Handle optional sign in scientific notation
                    if let Some(&next) = self.peek() {
                        if next == '+' || next == '-' {
                            number.push(self.advance().unwrap());
                        }
                    }
                }
            } else {
                break;
            }
        }

        if is_float {
            number.parse::<f64>()
                .map(Token::FloatLiteral)
                .map_err(|e| format!("Invalid float literal: {}", e))
        } else {
            // Try parsing as i64 first, if it fails, try as f64
            match number.parse::<i64>() {
                Ok(n) => Ok(Token::IntLiteral(n)),
                Err(_) => number.parse::<f64>()
                    .map(Token::FloatLiteral)
                    .map_err(|e| format!("Invalid number literal: {}", e))
            }
        }
    }

    /// Reads a string literal from the input.
    fn read_string(&mut self) -> Result<Token, String> {
        let mut string = String::new();
        while let Some(c) = self.advance() {
            if c == '"' {
                return Ok(Token::StringLiteral(string));
            }
            string.push(c);
        }
        Err("Unterminated string literal".to_string())
    }

    /// Recognizes and returns the next token in the input.
    pub fn next_token(&mut self) -> Result<Token, String> {
        self.skip_whitespace();

        match self.advance() {
            Some(c) => match c {
                'a'..='z' | 'A'..='Z' | '_' => {
                    let identifier = self.read_identifier(c);
                    match identifier.as_str() {
                        "let" => Ok(Token::Let),
                        "mut" => Ok(Token::Mut),
                        "func" => Ok(Token::Func),
                        "return" => Ok(Token::Return),
                        "if" => Ok(Token::If),
                        "else" => Ok(Token::Else),
                        "while" => Ok(Token::While),
                        "for" => Ok(Token::For),
                        "in" => Ok(Token::In),
                        "async" => Ok(Token::Async),
                        "await" => Ok(Token::Await),
                        "guard" => Ok(Token::Guard),
                        "int" => Ok(Token::Int),
                        "float" => Ok(Token::Float),
                        "string" => Ok(Token::String),
                        "bool" => Ok(Token::Bool),
                        "true" => Ok(Token::BoolLiteral(true)),
                        "false" => Ok(Token::BoolLiteral(false)),
                        _ => Ok(Token::Identifier(identifier)),
                    }
                }
                '0'..='9' => self.read_number(c),
                '"' => self.read_string(),
                '+' => Ok(Token::Plus),
                '-' => {
                    if self.peek() == Some(&'>') {
                        self.advance();
                        Ok(Token::Arrow)
                    } else {
                        Ok(Token::Minus)
                    }
                }
                '*' => Ok(Token::Multiply),
                '/' => Ok(Token::Divide),
                '=' => {
                    if self.peek() == Some(&'=') {
                        self.advance();
                        Ok(Token::Eq)
                    } else {
                        Ok(Token::Assign)
                    }
                }
                '!' => {
                    if self.peek() == Some(&'=') {
                        self.advance();
                        Ok(Token::NotEq)
                    } else {
                        Ok(Token::Not)
                    }
                }
                '<' => {
                    if self.peek() == Some(&'=') {
                        self.advance();
                        Ok(Token::LtEq)
                    } else {
                        Ok(Token::Lt)
                    }
                }
                '>' => {
                    if self.peek() == Some(&'=') {
                        self.advance();
                        Ok(Token::GtEq)
                    } else {
                        Ok(Token::Gt)
                    }
                }
                '&' => {
                    if self.peek() == Some(&'&') {
                        self.advance();
                        Ok(Token::And)
                    } else {
                        Err("Unexpected character: &".to_string())
                    }
                }
                '|' => {
                    if self.peek() == Some(&'|') {
                        self.advance();
                        Ok(Token::Or)
                    } else {
                        Err("Unexpected character: |".to_string())
                    }
                }
                '(' => Ok(Token::LParen),
                ')' => Ok(Token::RParen),
                '{' => Ok(Token::LBrace),
                '}' => Ok(Token::RBrace),
                '[' => Ok(Token::LBracket),
                ']' => Ok(Token::RBracket),
                ',' => Ok(Token::Comma),
                ':' => Ok(Token::Colon),
                ';' => Ok(Token::Semicolon),
                _ => Err(format!("Unexpected character: {}", c)),
            },
            None => Ok(Token::Eof),
        }
    }
}

/// Converts an input string into a vector of tokens.
pub fn tokenize(input: &str) -> Result<Vec<Token>, String> {
    let mut lexer = Lexer::new(input);
    let mut tokens = Vec::new();

    loop {
        match lexer.next_token() {
            Ok(token) => {
                if token == Token::Eof {
                    tokens.push(token);
                    break;
                }
                tokens.push(token);
            }
            Err(e) => return Err(e),
        }
    }

    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_tokens() {
        let input = "let x = 5;";
        let tokens = tokenize(input).unwrap();
        assert_eq!(tokens, vec![
            Token::Let,
            Token::Identifier("x".to_string()),
            Token::Assign,
            Token::IntLiteral(5),
            Token::Semicolon,
            Token::Eof
        ]);
    }

    #[test]
    fn test_complex_tokens() {
        let input = "func add(a: int, b: int) -> int { return a + b; }";
        let tokens = tokenize(input).unwrap();
        assert_eq!(tokens, vec![
            Token::Func,
            Token::Identifier("add".to_string()),
            Token::LParen,
            Token::Identifier("a".to_string()),
            Token::Colon,
            Token::Int,
            Token::Comma,
            Token::Identifier("b".to_string()),
            Token::Colon,
            Token::Int,
            Token::RParen,
            Token::Arrow,
            Token::Int,
            Token::LBrace,
            Token::Return,
            Token::Identifier("a".to_string()),
            Token::Plus,
            Token::Identifier("b".to_string()),
            Token::Semicolon,
            Token::RBrace,
            Token::Eof
        ]);
    }

    #[test]
    fn test_string_literal() {
        let input = r#"let greeting = "Hello, world!";"#;
        let tokens = tokenize(input).unwrap();
        assert_eq!(tokens, vec![
            Token::Let,
            Token::Identifier("greeting".to_string()),
            Token::Assign,
            Token::StringLiteral("Hello, world!".to_string()),
            Token::Semicolon,
            Token::Eof
        ]);
    }

    #[test]
    fn test_float_literal() {
        let input = "let pi = 3.141592653589793;";
        let tokens = tokenize(input).unwrap();
        assert_eq!(tokens.len(), 6);
        assert_eq!(tokens[0], Token::Let);
        assert_eq!(tokens[1], Token::Identifier("pi".to_string()));
        assert_eq!(tokens[2], Token::Assign);
        match &tokens[3] {
            Token::FloatLiteral(value) => {
                assert!((value - std::f64::consts::PI).abs() < f64::EPSILON);
            },
            _ => panic!("Expected FloatLiteral, got {:?}", tokens[3]),
        }
        assert_eq!(tokens[4], Token::Semicolon);
        assert_eq!(tokens[5], Token::Eof);
    }

    #[test]
    fn test_keywords_and_identifiers() {
        let input = "let mut while_loop = true;";
        let tokens = tokenize(input).unwrap();
        assert_eq!(tokens, vec![
            Token::Let,
            Token::Mut,
            Token::Identifier("while_loop".to_string()),
            Token::Assign,
            Token::BoolLiteral(true),
            Token::Semicolon,
            Token::Eof
        ]);
    }

    #[test]
    fn test_operators() {
        let input = "a + b - c * d / e == f != g < h > i <= j >= k && l || !m";
        let tokens = tokenize(input).unwrap();
        assert_eq!(tokens, vec![
            Token::Identifier("a".to_string()),
            Token::Plus,
            Token::Identifier("b".to_string()),
            Token::Minus,
            Token::Identifier("c".to_string()),
            Token::Multiply,
            Token::Identifier("d".to_string()),
            Token::Divide,
            Token::Identifier("e".to_string()),
            Token::Eq,
            Token::Identifier("f".to_string()),
            Token::NotEq,
            Token::Identifier("g".to_string()),
            Token::Lt,
            Token::Identifier("h".to_string()),
            Token::Gt,
            Token::Identifier("i".to_string()),
            Token::LtEq,
            Token::Identifier("j".to_string()),
            Token::GtEq,
            Token::Identifier("k".to_string()),
            Token::And,
            Token::Identifier("l".to_string()),
            Token::Or,
            Token::Not,
            Token::Identifier("m".to_string()),
            Token::Eof
        ]);
    }

    #[test]
    fn test_error_handling() {
        let input = "let x = 3.14.15;";
        assert!(tokenize(input).is_err());

        let input = "let y = \"unclosed string;";
        assert!(tokenize(input).is_err());

        let input = "let z = &invalid;";
        assert!(tokenize(input).is_err());
    }
}