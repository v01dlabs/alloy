use alloy::{
    error::LexerError,
    lexer::{Lexer, Token},
};

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
    let input = "fn add(a: int, b: int) -> int { return a + b; }";
    let tokens = Lexer::<'_>::tokenize(input).unwrap();
    assert_eq!(
        tokens,
        vec![
            Token::Fn,
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
    let input = "(a, b) { [c] } : ; . \n";
    let tokens = Lexer::tokenize(input).unwrap();
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
            Token::Newline,
            Token::Eof
        ]
    );
}

#[test]
fn test_newline_tokenization() {
    let input = "let x = 5\nlet y = 10\n\nfn test() {\n    print(x + y)\n}\n";
    let tokens = Lexer::tokenize(input).unwrap();
    let expected_tokens = vec![
        Token::Let,
        Token::Identifier("x".to_string()),
        Token::Assign,
        Token::IntLiteral(5),
        Token::Newline,
        Token::Let,
        Token::Identifier("y".to_string()),
        Token::Assign,
        Token::IntLiteral(10),
        Token::Newline,
        Token::Newline,
        Token::Fn,
        Token::Identifier("test".to_string()),
        Token::LParen,
        Token::RParen,
        Token::LBrace,
        Token::Newline,
        Token::Identifier("print".to_string()),
        Token::LParen,
        Token::Identifier("x".to_string()),
        Token::Plus,
        Token::Identifier("y".to_string()),
        Token::RParen,
        Token::Newline,
        Token::RBrace,
        Token::Newline,
        Token::Eof,
    ];
    assert_eq!(tokens, expected_tokens);
}

#[test]
fn test_arrow_operator() {
    let input = "fn foo() -> int {}";
    let tokens = Lexer::<'_>::tokenize(input).unwrap();
    assert_eq!(
        tokens,
        vec![
            Token::Fn,
            Token::Identifier("foo".to_string()),
            Token::LParen,
            Token::RParen,
            Token::Arrow,
            Token::Identifier("int".to_string()),
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
        Err(LexerError::InvalidNumber(_, _))
    ));

    let input = "let y = $100;";
    assert!(matches!(
        Lexer::tokenize(input),
        Err(LexerError::UnexpectedChar('$', _))
    ));
}

#[test]
fn test_error_handling() {
    let input = "let x = 3.14.15;";
    assert!(matches!(
        Lexer::tokenize(input),
        Err(LexerError::InvalidNumber(_, _))
    ));

    let input = "let y = @invalid;";
    assert!(matches!(
        Lexer::tokenize(input),
        Err(LexerError::UnexpectedChar('@', _))
    ));
}
