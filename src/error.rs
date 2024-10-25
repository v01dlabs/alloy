use crate::lexer::Location;
use miette::{Diagnostic, SourceSpan};
use owo_colors::{DynColors, OwoColorize};
use std::error::Error;
use std::fmt;

const ALLOY_YELLOW: DynColors = DynColors::Rgb(222, 165, 132);
const ALLOY_ORANGE: DynColors = DynColors::Rgb(240, 81, 56);
const _ALLOY_BLUE: DynColors = DynColors::Rgb(49, 120, 198);

#[derive(Debug, Diagnostic)]
pub enum CompilerError {
    #[diagnostic(code(alloy::lexer_error))]
    LexerError(LexerError),

    #[diagnostic(code(alloy::parser_error))]
    ParserError(ParserError),
    #[diagnostic(code(alloy::type_error))]
    TypeError(TypeError),   
}

impl fmt::Display for CompilerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompilerError::LexerError(e) => {
                write!(f, "{}", format!("Lexer error: {}", e).color(ALLOY_YELLOW))
            }
            CompilerError::ParserError(e) => {
                write!(f, "{}", format!("Parser error: {}", e).color(ALLOY_ORANGE))
            },
            CompilerError::TypeError(e) => {
                write!(f, "{}", format!("Type error: {}", e).color(ALLOY_ORANGE))
            }
        }
    }
}

impl Error for CompilerError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            CompilerError::LexerError(e) => Some(e),
            CompilerError::ParserError(e) => Some(e),
            CompilerError::TypeError(e) => Some(e),
        }
    }
}

#[derive(Debug, Diagnostic)]
pub enum LexerError {
    #[diagnostic(code(alloy::lexer::unexpected_char))]
    UnexpectedChar(char, #[label("Unexpected character found here")] SourceSpan),

    #[diagnostic(code(alloy::lexer::invalid_number))]
    InvalidNumber(String, #[label("Invalid number found here")] SourceSpan),

    #[diagnostic(code(alloy::lexer::unterminated_string))]
    UnterminatedString(#[label("Unterminated string starts here")] SourceSpan),
}

impl LexerError {
    pub fn to_miette_span(location: &Location) -> SourceSpan {
        SourceSpan::new((location.line * 1000 + location.column).into(), 1_usize)
    }
}

impl fmt::Display for LexerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LexerError::UnexpectedChar(c, _) => write!(
                f,
                "{}",
                format!("Unexpected character: {}", c).color(ALLOY_YELLOW)
            ),
            LexerError::InvalidNumber(s, _) => write!(
                f,
                "{}",
                format!("Invalid number: {}", s).color(ALLOY_YELLOW)
            ),
            LexerError::UnterminatedString(_) => {
                write!(f, "{}", "Unterminated string".color(ALLOY_YELLOW))
            }
        }
    }
}

impl Error for LexerError {}

#[derive(Debug, Diagnostic)]
pub enum ParserError {
    #[diagnostic(code(alloy::parser::unexpected_token))]
    UnexpectedToken(String),

    #[diagnostic(code(alloy::parser::expected_token))]
    ExpectedToken(String, String),

    #[diagnostic(code(alloy::parser::invalid_expression))]
    InvalidExpression,
}

impl fmt::Display for ParserError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParserError::UnexpectedToken(msg) => write!(
                f,
                "{}",
                format!("Unexpected token: {}", msg).color(ALLOY_ORANGE)
            ),
            ParserError::ExpectedToken(expected, found) => {
                write!(
                    f,
                    "{}",
                    format!("Expected {}, found {}", expected, found).color(ALLOY_ORANGE)
                )
            }
            ParserError::InvalidExpression => {
                write!(f, "{}", "Invalid expression".color(ALLOY_ORANGE))
            }
        }
    }
}

impl Error for ParserError {}


/// Represents a typing error.
#[derive(Debug, thiserror::Error)]
pub struct TypeError {
    pub message: String,
}

impl std::fmt::Display for TypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Type Error: {}", self.message)
    }
}
