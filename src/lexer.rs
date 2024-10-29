//! Lexical analysis for Alloy
//!
//! The lexer is responsible for converting raw source code into
//! a series of tokens that can be processed by the parser.

use crate::error::LexerError;
use std::fmt;
use std::iter::Peekable;
use std::str::Chars;

/// Represents a token in Alloy.
#[derive(Debug, PartialEq, Clone)]
pub enum Token {
    // Keywords
    Let,
    Run,
    Return,
    Fn,
    If,

    Else,
    While,
    Loop,
    Do, // I really miss these in Rust sometimes
    For,
    Match,
    Guard,

    Mut,
    Async,
    Shared,
    Default,
    In,
    With,
    Where,

    Await,

    Struct, // ? unsure if keeping
    Enum,   // ? unsure if keeping
    Union,
    Type,
    Effect,
    Trait,
    Handler,

    Impl,

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
    Range,

    Typeof,
    As,

    // Delimiters
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Pipe,
    Comma,
    Dot,
    QuestionMark,
    ExclamationPt,
    Colon,
    PathSep,
    Arrow,
    Semicolon,
    Newline,

    // Special
    Eof,
}

impl Token {
    pub fn is_block_end(&self) -> bool {
        matches!(
            self,
            Token::RBrace | Token::RBracket | Token::RParen | Token::Eof
        )
    }

    pub fn is_block_start(&self) -> bool {
        matches!(self, Token::LBrace | Token::LBracket | Token::LParen)
            || matches!(
                self,
                Token::If
                    | Token::Let
                    | Token::Else
                    | Token::For
                    | Token::While
                    | Token::Guard
                    | Token::Match
            )
    }

    pub fn ident_to_keyword<'a>(&'a self) -> &'a Token {
        match self {
            Token::Identifier(ref ident) => match ident.as_str() {
                "handler" => &Token::Handler,
                "shared" => &Token::Shared,
                "default" => &Token::Default,
                "effect" => &Token::Effect,
                _ => self,
            },
            _ => self,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Location {
    pub line: usize,
    pub column: usize,
}

impl Location {
    pub fn n(line: usize, column: usize) -> Location {
        Location { line, column }
    }
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.line, self.column)
    }
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

    /// Returns the current line and column position of the lexer in a tuple struct.
    /// Primarily for more useful error production.
    #[inline]
    fn loc(&self) -> Location {
        Location::n(self.line, self.column)
    }

    /// Peeks at the next character in the input without advancing the lexer.
    fn peek(&mut self) -> Option<&char> {
        self.input.peek()
    }

    /// Skips over any whitespace characters in the input.
    fn skip_whitespace(&mut self) {
        while let Some(&c) = self.peek() {
            if c.is_whitespace() && c != '\n' {
                self.advance();
            } else {
                break;
            }
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
                return Err(LexerError::InvalidNumber(
                    number,
                    LexerError::to_miette_span(&self.loc()),
                ));
            } else {
                break;
            }
        }

        if has_decimal {
            number.parse::<f64>().map(Token::FloatLiteral).map_err(|_| {
                LexerError::InvalidNumber(number, LexerError::to_miette_span(&self.loc()))
            })
        } else {
            number.parse::<i64>().map(Token::IntLiteral).map_err(|_| {
                LexerError::InvalidNumber(number, LexerError::to_miette_span(&self.loc()))
            })
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
        Err(LexerError::UnterminatedString(LexerError::to_miette_span(
            &self.loc(),
        )))
    }

    /// Matches an identifier to a keyword token.
    fn match_keyword(ident: &str) -> Token {
        match ident {
            "let" => Token::Let,
            "fn" => Token::Fn,
            "if" => Token::If,
            "else" => Token::Else,
            "while" => Token::While,
            "loop" => Token::Loop,
            // "do" => Token::Do,
            "for" => Token::For,
            "return" => Token::Return,
            "guard" => Token::Guard,
            "mut" => Token::Mut,
            "in" => Token::In,
            "async" => Token::Async,
            "await" => Token::Await,
            "true" => Token::BoolLiteral(true),
            "false" => Token::BoolLiteral(false),
            "match" => Token::Match,
            "effect" => Token::Effect,
            "type" => Token::Type,
            "struct" => Token::Struct,
            "enum" => Token::Enum,
            "union" => Token::Union,
            "trait" => Token::Trait,
            "impl" => Token::Impl,
            "handler" => Token::Handler,
            "typeof" => Token::Typeof,
            "as" => Token::As,
            "default" => Token::Default,
            "shared" => Token::Shared,
            "run" => Token::Run,
            "with" => Token::With,

            "\r\n" => Token::Newline, // windows line ending bit awakward here but...
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
                    let next = self.peek();
                    if next == Some(&'=') {
                        self.advance();
                        Ok(Token::NotEq)
                    } else if next == Some(&'.') {
                        Ok(Token::ExclamationPt)
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
                        // consider having bitwise operators
                        Err(LexerError::UnexpectedChar(
                            '&',
                            LexerError::to_miette_span(&self.loc()),
                        ))
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
                        Ok(Token::Pipe)
                    }
                }
                '(' => Ok(Token::LParen),
                ')' => Ok(Token::RParen),
                '{' => Ok(Token::LBrace),
                '}' => Ok(Token::RBrace),
                '[' => Ok(Token::LBracket),
                ']' => Ok(Token::RBracket),
                ',' => Ok(Token::Comma),
                '.' => {
                    let next = self.peek();
                    if next == Some(&'.') {
                        self.advance();
                        Ok(Token::Range)
                    } else {
                        Ok(Token::Dot)
                    }
                }
                ':' => {
                    let next = self.peek();
                    if next == Some(&':') {
                        self.advance();
                        Ok(Token::PathSep)
                    } else {
                        Ok(Token::Colon)
                    }
                }
                ';' => Ok(Token::Semicolon),
                '?' => Ok(Token::QuestionMark),
                '\n' => Ok(Token::Newline),

                _ => Err(LexerError::UnexpectedChar(
                    c,
                    LexerError::to_miette_span(&self.loc()),
                )),
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
