//! Lexical analysis for Alloy
//!
//! The lexer is responsible for converting raw source code into
//! a series of tokens that can be processed by the parser.

use crate::error::LexerError;
use std::iter::Peekable;
use std::str::Chars;

/// Represents a token in Alloy.
#[derive(Debug, PartialEq, Clone)]
pub enum Token {
    // Keywords
    Let,
    Func,
    If,
    Else,
    While,
    For,
    Return,
    Guard,
    Mut,
    In,
    Async,
    Await,
    Match,

    // Types
    Int,
    Float,
    String,
    Bool,

    // Literals
    Identifier(String),
    IntLiteral(i64),
    FloatLiteral(f64),
    StringLiteral(String),
    BoolLiteral(bool),

    // Operators
    Plus,
    Minus,
    Multiply,
    Divide,
    Assign,
    Eq,
    Modulo,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    And,
    Or,
    Not,
    Pipeline,
    Increment,
    Decrement,

    // Delimiters
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Comma,
    Dot,
    Colon,
    Arrow,
    Semicolon,

    // Special
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
    fn read_number(&mut self, first_char: char) -> Result<Token, LexerError> {
        let mut number = String::new();
        number.push(first_char);
        let mut has_decimal = false;

        while let Some(&c) = self.peek() {
            if c.is_ascii_digit() {
                number.push(self.advance().unwrap());
            } else if c == '.' && !has_decimal {
                has_decimal = true;
                number.push(self.advance().unwrap());
            } else if c == '.' && has_decimal {
                // We've encountered a second decimal point, which is invalid
                return Err(LexerError::InvalidNumber(number));
            } else {
                break;
            }
        }

        if has_decimal {
            number
                .parse::<f64>()
                .map(Token::FloatLiteral)
                .map_err(|_| LexerError::InvalidNumber(number))
        } else {
            number
                .parse::<i64>()
                .map(Token::IntLiteral)
                .map_err(|_| LexerError::InvalidNumber(number))
        }
    }

    /// Reads a string literal from the input.
    fn read_string(&mut self) -> Result<Token, LexerError> {
        let mut string = String::new();
        while let Some(c) = self.advance() {
            if c == '"' {
                return Ok(Token::StringLiteral(string));
            }
            string.push(c);
        }
        Err(LexerError::UnterminatedString)
    }

    // Update the `match_keyword` function in the lexer
    fn match_keyword(ident: &str) -> Token {
        match ident {
            "let" => Token::Let,
            "func" => Token::Func,
            "if" => Token::If,
            "else" => Token::Else,
            "while" => Token::While,
            "for" => Token::For,
            "return" => Token::Return,
            "guard" => Token::Guard,
            "mut" => Token::Mut,
            "in" => Token::In,
            "async" => Token::Async,
            "await" => Token::Await,
            "int" => Token::Int,
            "float" => Token::Float,
            "string" => Token::String,
            "bool" => Token::Bool,
            "true" => Token::BoolLiteral(true),
            "false" => Token::BoolLiteral(false),
            "match" => Token::Match,
            _ => Token::Identifier(ident.to_string()),
        }
    }

    /// Recognizes and returns the next token in the input.
    pub fn next_token(&mut self) -> Result<Token, LexerError> {
        self.skip_whitespace();
        match self.advance() {
            Some(c) => match c {
                '0'..='9' => self.read_number(c),
                'a'..='z' | 'A'..='Z' | '_' => {
                    let identifier = self.read_identifier(c);
                    Ok(Self::match_keyword(&identifier))
                }
                '"' => self.read_string(),
                '+' => {
                    if self.peek() == Some(&'+') {
                        self.advance();
                        Ok(Token::Increment)
                    } else {
                        Ok(Token::Plus)
                    }
                }
                '-' => {
                    if self.peek() == Some(&'-') {
                        self.advance();
                        Ok(Token::Decrement)
                    } else if self.peek() == Some(&'>') {
                        self.advance();
                        Ok(Token::Arrow)
                    } else {
                        Ok(Token::Minus)
                    }
                }
                '*' => Ok(Token::Multiply),
                '/' => Ok(Token::Divide),
                '%' => Ok(Token::Modulo),
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
                        Err(LexerError::UnexpectedChar('&'))
                    }
                }
                '|' => {
                    if self.peek() == Some(&'>') {
                        self.advance();
                        Ok(Token::Pipeline)
                    } else if self.peek() == Some(&'|') {
                        self.advance();
                        Ok(Token::Or)
                    } else {
                        Err(LexerError::UnexpectedChar('|'))
                    }
                }
                '(' => Ok(Token::LParen),
                ')' => Ok(Token::RParen),
                '{' => Ok(Token::LBrace),
                '}' => Ok(Token::RBrace),
                '[' => Ok(Token::LBracket),
                ']' => Ok(Token::RBracket),
                ',' => Ok(Token::Comma),
                '.' => Ok(Token::Dot),
                ':' => Ok(Token::Colon),
                ';' => Ok(Token::Semicolon),
                _ => Err(LexerError::UnexpectedChar(c)),
            },
            None => Ok(Token::Eof),
        }
    }

    /// Converts an input string into a vector of tokens.
    pub fn tokenize(input: &str) -> Result<Vec<Token>, LexerError> {
        let mut lexer = Lexer::new(input);
        let mut tokens = Vec::new();

        loop {
            match lexer.next_token() {
                Ok(Token::Eof) => {
                    tokens.push(Token::Eof);
                    break;
                }
                Ok(token) => tokens.push(token),
                Err(e) => return Err(e),
            }
        }

        Ok(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_tokens() {
        let input = "let x = 5;";
        let tokens = Lexer::<'_>::tokenize(input).unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Let,
                Token::Identifier("x".to_string()),
                Token::Assign,
                Token::IntLiteral(5),
                Token::Semicolon,
                Token::Eof
            ]
        );
    }

    #[test]
    fn test_complex_tokens() {
        let input = "func add(a: int, b: int) -> int { return a + b; }";
        let tokens = Lexer::<'_>::tokenize(input).unwrap();
        assert_eq!(
            tokens,
            vec![
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
            ]
        );
    }

    #[test]
    fn test_string_literal() {
        let input = r#"let greeting = "Hello, world!";"#;
        let tokens = Lexer::<'_>::tokenize(input).unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Let,
                Token::Identifier("greeting".to_string()),
                Token::Assign,
                Token::StringLiteral("Hello, world!".to_string()),
                Token::Semicolon,
                Token::Eof
            ]
        );
    }

    #[test]
    fn test_float_literal() {
        let input = "let pi = 3.141592653589793;";
        let tokens = Lexer::<'_>::tokenize(input).unwrap();
        assert_eq!(tokens.len(), 6);
        assert_eq!(tokens[0], Token::Let);
        assert_eq!(tokens[1], Token::Identifier("pi".to_string()));
        assert_eq!(tokens[2], Token::Assign);
        match &tokens[3] {
            Token::FloatLiteral(value) => {
                assert!((value - std::f64::consts::PI).abs() < f64::EPSILON);
            }
            _ => panic!("Expected FloatLiteral, got {:?}", tokens[3]),
        }
        assert_eq!(tokens[4], Token::Semicolon);
        assert_eq!(tokens[5], Token::Eof);
    }

    #[test]
    fn test_keywords_and_identifiers() {
        let input = "let mut while_loop = true;";
        let tokens = Lexer::<'_>::tokenize(input).unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Let,
                Token::Mut,
                Token::Identifier("while_loop".to_string()),
                Token::Assign,
                Token::BoolLiteral(true),
                Token::Semicolon,
                Token::Eof
            ]
        );
    }

    #[test]
    fn test_operators() {
        let input = "a + b - c * d / e % f == g != h < i > j <= k >= l && m || !n";
        let tokens = Lexer::tokenize(input).unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Identifier("a".to_string()),
                Token::Plus,
                Token::Identifier("b".to_string()),
                Token::Minus,
                Token::Identifier("c".to_string()),
                Token::Multiply,
                Token::Identifier("d".to_string()),
                Token::Divide,
                Token::Identifier("e".to_string()),
                Token::Modulo,
                Token::Identifier("f".to_string()),
                Token::Eq,
                Token::Identifier("g".to_string()),
                Token::NotEq,
                Token::Identifier("h".to_string()),
                Token::Lt,
                Token::Identifier("i".to_string()),
                Token::Gt,
                Token::Identifier("j".to_string()),
                Token::LtEq,
                Token::Identifier("k".to_string()),
                Token::GtEq,
                Token::Identifier("l".to_string()),
                Token::And,
                Token::Identifier("m".to_string()),
                Token::Or,
                Token::Not,
                Token::Identifier("n".to_string()),
                Token::Eof
            ]
        );
    }

    #[test]
    fn test_increment_operator() {
        let input = "let x = 5; x++;";
        let tokens = Lexer::<'_>::tokenize(input).unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Let,
                Token::Identifier("x".to_string()),
                Token::Assign,
                Token::IntLiteral(5),
                Token::Semicolon,
                Token::Identifier("x".to_string()),
                Token::Increment,
                Token::Semicolon,
                Token::Eof
            ]
        );
    }

    #[test]
    fn test_decrement_operator() {
        let input = "let x = 5; x--;";
        let tokens = Lexer::<'_>::tokenize(input).unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Let,
                Token::Identifier("x".to_string()),
                Token::Assign,
                Token::IntLiteral(5),
                Token::Semicolon,
                Token::Identifier("x".to_string()),
                Token::Decrement,
                Token::Semicolon,
                Token::Eof
            ]
        );
    }

    #[test]
    fn test_logical_operators() {
        let input = "a && b || !c;";
        let tokens = Lexer::tokenize(input).unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Identifier("a".to_string()),
                Token::And,
                Token::Identifier("b".to_string()),
                Token::Or,
                Token::Not,
                Token::Identifier("c".to_string()),
                Token::Semicolon,
                Token::Eof
            ]
        );
    }

    #[test]
    fn test_comparison_operators() {
        let input = "a == b != c < d <= e > f >= g;";
        let tokens = Lexer::<'_>::tokenize(input).unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Identifier("a".to_string()),
                Token::Eq,
                Token::Identifier("b".to_string()),
                Token::NotEq,
                Token::Identifier("c".to_string()),
                Token::Lt,
                Token::Identifier("d".to_string()),
                Token::LtEq,
                Token::Identifier("e".to_string()),
                Token::Gt,
                Token::Identifier("f".to_string()),
                Token::GtEq,
                Token::Identifier("g".to_string()),
                Token::Semicolon,
                Token::Eof
            ]
        );
    }

    #[test]
    fn test_arithmetic_operators() {
        let input = "a + b - c * d / e % f;";
        let tokens = Lexer::tokenize(input).unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Identifier("a".to_string()),
                Token::Plus,
                Token::Identifier("b".to_string()),
                Token::Minus,
                Token::Identifier("c".to_string()),
                Token::Multiply,
                Token::Identifier("d".to_string()),
                Token::Divide,
                Token::Identifier("e".to_string()),
                Token::Modulo,
                Token::Identifier("f".to_string()),
                Token::Semicolon,
                Token::Eof
            ]
        );
    }

    #[test]
    fn test_pipeline_operator() {
        let input = "a |> b;";
        let tokens = Lexer::<'_>::tokenize(input).unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Identifier("a".to_string()),
                Token::Pipeline,
                Token::Identifier("b".to_string()),
                Token::Semicolon,
                Token::Eof
            ]
        );
    }

    #[test]
    fn test_delimiters() {
        let input = "(a, b) { [c] } : ; .";
        let tokens = Lexer::<'_>::tokenize(input).unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::LParen,
                Token::Identifier("a".to_string()),
                Token::Comma,
                Token::Identifier("b".to_string()),
                Token::RParen,
                Token::LBrace,
                Token::LBracket,
                Token::Identifier("c".to_string()),
                Token::RBracket,
                Token::RBrace,
                Token::Colon,
                Token::Semicolon,
                Token::Dot,
                Token::Eof
            ]
        );
    }

    #[test]
    fn test_arrow_operator() {
        let input = "func foo() -> int {}";
        let tokens = Lexer::<'_>::tokenize(input).unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Func,
                Token::Identifier("foo".to_string()),
                Token::LParen,
                Token::RParen,
                Token::Arrow,
                Token::Int,
                Token::LBrace,
                Token::RBrace,
                Token::Eof
            ]
        );
    }

    #[test]
    fn test_unclosed_string_literal() {
        let input = "let x = \"unclosed string;";
        assert!(Lexer::<'_>::tokenize(input).is_err());
    }

    #[test]
    fn test_invalid_character() {
        let input = "let x = &invalid;";
        assert!(Lexer::<'_>::tokenize(input).is_err());
    }

    #[test]
    fn test_unexpected_character() {
        let input = "let x = 3.14.15;";
        assert!(matches!(
            Lexer::tokenize(input),
            Err(LexerError::InvalidNumber(_))
        ));

        let input = "let y = $100;";
        assert!(matches!(
            Lexer::tokenize(input),
            Err(LexerError::UnexpectedChar('$'))
        ));
    }

    #[test]
    fn test_error_handling() {
        let input = "let x = 3.14.15;";
        assert!(matches!(
            Lexer::tokenize(input),
            Err(LexerError::InvalidNumber(_))
        ));

        let input = "let y = @invalid;";
        assert!(matches!(
            Lexer::tokenize(input),
            Err(LexerError::UnexpectedChar('@'))
        ));
    }
}
