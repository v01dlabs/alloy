use std::error::Error;
use std::fmt;

#[derive(Debug)]
pub enum CompilerError {
    LexerError(LexerError),
    ParserError(ParserError),
}

impl fmt::Display for CompilerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompilerError::LexerError(e) => write!(f, "Lexer error: {}", e),
            CompilerError::ParserError(e) => write!(f, "Parser error: {}", e),
        }
    }
}

impl Error for CompilerError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            CompilerError::LexerError(e) => Some(e),
            CompilerError::ParserError(e) => Some(e),
        }
    }
}

#[derive(Debug)]
pub enum LexerError {
    UnexpectedChar(char),
    InvalidNumber(String),
    UnterminatedString,
}

impl fmt::Display for LexerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LexerError::UnexpectedChar(c) => write!(f, "Unexpected character: {}", c),
            LexerError::InvalidNumber(s) => write!(f, "Invalid number: {}", s),
            LexerError::UnterminatedString => write!(f, "Unterminated string"),
        }
    }
}

impl Error for LexerError {}

#[derive(Debug)]
pub enum ParserError {
    UnexpectedToken(String),
    ExpectedToken(String, String),
    InvalidExpression,
}

impl fmt::Display for ParserError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParserError::UnexpectedToken(msg) => write!(f, "Unexpected token: {}", msg),
            ParserError::ExpectedToken(expected, found) => {
                write!(f, "Expected {}, found {}", expected, found)
            }
            ParserError::InvalidExpression => write!(f, "Invalid expression"),
        }
    }
}

impl Error for ParserError {}
