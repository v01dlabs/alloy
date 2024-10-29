#![feature(box_patterns)]

use alloy::{
    ast::{
        ty::{
            FnRetTy, Function, GenericParam, GenericParamKind, Mutability, Param, Path, Pattern,
            Ty, TyKind,
        },
        AstElem, BinaryOperator, BindAttr, Expr, FnAttr, ImplKind, Item, Literal, Precedence,
        Statement, UnaryOperator, WithClauseItem, P,
    },
    error::ParserError,
    lexer::{token::Token, Lexer},
    parser::{parse, Parser},
};
use thin_vec::thin_vec;

fn init_tracing() {
    let format = tracing_subscriber::fmt::format().pretty();

    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .event_format(format)
        .with_test_writer()
        .with_ansi(true)
        .try_init();
}

// Helper function to create a parser from a vector of tokens
fn create_parser(tokens: Vec<Token>) -> Parser {
    init_tracing();
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
    let result = parser.parse_declaration();
    assert!(
        result.is_ok(),
        "Failed to parse variable declaration: {}",
        result.unwrap_err()
    );
    let expected = P(AstElem::item(P(Item::bind(
        "x".to_string(),
        thin_vec![BindAttr {
            mutability: Mutability::Not,
            ref_kind: None
        }],
        Some(P(Ty::simple("int".to_string()))),
        Some(P(Expr::literal(Literal::Int(5)))),
    ))));
    assert_eq!(result.unwrap(), expected);
}

#[test]
fn test_parse_unary_operator() {
    let tokens = vec![
        Token::Not,
        Token::Identifier("x".to_string()),
        Token::Semicolon,
    ];
    let mut parser = create_parser(tokens);
    let result = parser.parse_declaration().unwrap();
    let expected = P(AstElem::statement(P(Statement::expr(P(Expr::unary(
        UnaryOperator::Not,
        P(Expr::path(None, Path::ident("x".to_string()))),
    ))))));
    assert_eq!(result, expected);
}

#[test]
fn test_parse_function_declaration() {
    let tokens = vec![
        Token::Fn,
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
    let expected = P(AstElem::item(P(Item::fn_(
        "add".to_string(),
        thin_vec![],
        Function {
            generic_params: thin_vec![GenericParam::simple("T".to_string()),],
            inputs: thin_vec![
                Param {
                    name: "a".to_string(),
                    ty: P(Ty::simple("T".to_string())),
                },
                Param {
                    name: "b".to_string(),
                    ty: P(Ty::simple("T".to_string())),
                },
            ],
            output: FnRetTy::Ty(P(Ty::simple("T".to_string()))),
        },
        thin_vec![P(Statement::expr(P(Expr::return_(Some(P(Expr::binary(
            BinaryOperator::Add,
            P(Expr::path(None, Path::ident("a".to_string()))),
            P(Expr::path(None, Path::ident("b".to_string()))),
        ))),))))],
    ))));
    assert_eq!(result, expected);
}

#[test]
fn test_parse_multiple_declarations() {
    let tokens = vec![
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
        Token::Eof,
    ];
    let mut parser = create_parser(tokens);
    let result = parser.parse();
    assert!(result.is_ok());
    let expected = P(AstElem::program(thin_vec![
        P(AstElem::item(P(Item::bind(
            "x".to_string(),
            thin_vec![BindAttr {
                mutability: Mutability::Not,
                ref_kind: None
            }],
            None,
            Some(P(Expr::literal(Literal::Int(5)))),
        )))),
        P(AstElem::item(P(Item::bind(
            "y".to_string(),
            thin_vec![BindAttr {
                mutability: Mutability::Not,
                ref_kind: None
            }],
            None,
            Some(P(Expr::literal(Literal::Int(10)))),
        )))),
    ]));
    assert_eq!(result.unwrap(), expected);
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
    let expected = P(Statement::expr(P(Expr::if_(
        P(Expr::binary(
            BinaryOperator::GreaterThan,
            P(Expr::path(None, Path::ident("x".to_string()))),
            P(Expr::literal(Literal::Int(5))),
        )),
        P(Expr::block(
            thin_vec![P(Statement::expr(P(Expr::return_(Some(P(
                Expr::literal(Literal::Bool(true))
            )),))))],
            None,
        )),
        Some(P(Expr::block(
            thin_vec![P(Statement::expr(P(Expr::return_(Some(P(
                Expr::literal(Literal::Bool(false))
            )),))))],
            None,
        ))),
    ))));
    assert_eq!(result, expected);
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
    let result = parser.parse_statement();
    assert!(
        result.is_ok(),
        "Failed to parse while loop: {}",
        result.unwrap_err()
    );
    let expected = P(Statement::expr(P(Expr::while_(
        P(Expr::binary(
            BinaryOperator::LessThan,
            P(Expr::path(None, Path::ident("i".to_string()))),
            P(Expr::literal(Literal::Int(10))),
        )),
        P(Expr::block(
            thin_vec![P(Statement::expr(P(Expr::assign(
                P(Expr::path(None, Path::ident("i".to_string()))),
                P(Expr::binary(
                    BinaryOperator::Add,
                    P(Expr::path(None, Path::ident("i".to_string()))),
                    P(Expr::literal(Literal::Int(1))),
                )),
            )))),],
            None,
        )),
        None,
    ))));
    assert_eq!(result.unwrap(), expected);
}

#[test]
fn test_parse_for_statement() {
    let source = "for name in names { print(name) }";
    let tokens = Lexer::tokenize(source).unwrap();
    let mut parser = create_parser(tokens);
    let result = parser.parse_statement();
    assert!(
        result.is_ok(),
        "Failed to parse for statement: {}",
        result.unwrap_err()
    );
    let expected = P(Statement::expr(P(Expr::for_(
        P(Pattern::id_simple("name".to_string())),
        P(Expr::path(None, Path::ident("names".to_string()))),
        P(Expr::block(
            thin_vec![P(Statement::expr(P(Expr::call(
                P(Expr::path(None, Path::ident("print".to_string()))),
                None,
                thin_vec![P(Expr::path(None, Path::ident("name".to_string())))],
            ))))],
            None,
        )),
        None,
    ))));
    assert_eq!(result.unwrap(), expected);
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
    let expected = P(Statement::expr(P(Expr::pipeline(
        P(AstElem::expr(P(Expr::pipeline(
            P(AstElem::expr(P(Expr::path(
                None,
                Path::ident("x".to_string()),
            )))),
            P(Expr::call(
                P(Expr::path(None, Path::ident("foo".to_string()))),
                None,
                thin_vec![],
            )),
        )))),
        P(Expr::call(
            P(Expr::path(None, Path::ident("bar".to_string()))),
            None,
            thin_vec![],
        )),
    ))));
    assert_eq!(result, expected);
}

#[test]
fn test_parse_trailing_closure() {
    let source = "someFunction() { in return 42 }";
    let tokens = Lexer::tokenize(source).unwrap();
    println!("{:?}", tokens);
    let mut parser = create_parser(tokens);
    let result = parser.parse_expression(Precedence::None);
    assert!(
        result.is_ok(),
        "Failed to parse trailing closure: {}",
        result.unwrap_err()
    );
    let expected = P(Expr::trailing_closure(
        P(Expr::call(
            P(Expr::path(None, Path::ident("someFunction".to_string()))),
            None,
            thin_vec![],
        )),
        thin_vec![],
        P(Expr::block(
            thin_vec![P(Statement::expr(P(Expr::return_(Some(P(
                Expr::literal(Literal::Int(42))
            )),))))],
            None,
        )),
    ));
    assert_eq!(result.unwrap(), expected);
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
    // assert!(matches!(result,
    //     box AstNode::GuardStatement {
    //         condition: box AstNode::BinaryOperation { .. },
    //         body: box AstNode::Block(statements)
    //     } if statements.len() == 1
    // ));
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
    let result = parser.parse_statement();
    println!("{:?}", result);
    assert!(
        result.is_ok(),
        "Failed to parse return statement: {}",
        result.unwrap_err()
    );
    let expected = P(Statement::expr(P(Expr::return_(Some(P(Expr::binary(
        BinaryOperator::Add,
        P(Expr::path(None, Path::ident("x".to_string()))),
        P(Expr::path(None, Path::ident("y".to_string()))),
    )))))));
    assert_eq!(result.unwrap(), expected);
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
    let expected = P(Statement::expr(P(Expr::binary(
        BinaryOperator::Add,
        P(Expr::path(None, Path::ident("a".to_string()))),
        P(Expr::binary(
            BinaryOperator::Multiply,
            P(Expr::path(None, Path::ident("b".to_string()))),
            P(Expr::binary(
                BinaryOperator::Subtract,
                P(Expr::path(None, Path::ident("c".to_string()))),
                P(Expr::path(None, Path::ident("d".to_string()))),
            )),
        )),
    ))));
    assert_eq!(result, expected);
}

#[test]
fn test_basic_effect() {
    let input = r#"
    effect IO {
        read_line() -> String
        print(str: String)
    }
    "#;
    let tokens = Lexer::tokenize(input).unwrap();
    let mut parser = create_parser(tokens);
    let result = parser.parse();
    assert!(
        result.is_ok(),
        "Failed to parse effect declaration: {}",
        result.unwrap_err()
    );
    let expected = P(AstElem::program(thin_vec![P(AstElem::item(P(
        Item::effect(
            "IO".to_string(),
            thin_vec![],
            None,
            thin_vec![],
            thin_vec![
                P(AstElem::item(P(Item::fn_(
                    "read_line".to_string(),
                    thin_vec![],
                    Function {
                        generic_params: thin_vec![],
                        inputs: thin_vec![],
                        output: FnRetTy::Ty(P(Ty::simple("String".to_string()))),
                    },
                    thin_vec![],
                )))),
                P(AstElem::item(P(Item::fn_(
                    "print".to_string(),
                    thin_vec![],
                    Function {
                        generic_params: thin_vec![],
                        inputs: thin_vec![Param {
                            name: "str".to_string(),
                            ty: P(Ty::simple("String".to_string())),
                        },],
                        output: FnRetTy::default(),
                    },
                    thin_vec![],
                )))),
            ],
        )
    ))),]));
    assert_eq!(result.unwrap(), expected);
}

#[test]
fn test_struct_decl() {
    let input = r#"
    struct List[T] {
        head: Option[T]
        tail: Option[Box[List[T]]]
    }
    "#;
    let tokens = Lexer::tokenize(input).unwrap();
    let mut parser = create_parser(tokens);
    let result = parser.parse();
    assert!(
        result.is_ok(),
        "Failed to parse struct declaration: {}",
        result.unwrap_err()
    );
    let expected = P(AstElem::program(thin_vec![P(AstElem::item(P(
        Item::struct_(
            "List".to_string(),
            thin_vec![GenericParam::simple("T".to_string()),],
            thin_vec![],
            thin_vec![
                P(AstElem::item(P(Item::bind(
                    "head".to_string(),
                    thin_vec![BindAttr {
                        mutability: Mutability::Not,
                        ref_kind: None
                    }],
                    Some(P(Ty::generic(
                        "Option".to_string(),
                        thin_vec![P(Ty::simple("T".to_string()))]
                    ))),
                    None,
                )))),
                P(AstElem::item(P(Item::bind(
                    "tail".to_string(),
                    thin_vec![BindAttr {
                        mutability: Mutability::Not,
                        ref_kind: None
                    }],
                    Some(P(Ty::generic(
                        "Option".to_string(),
                        thin_vec![P(Ty::generic(
                            "Box".to_string(),
                            thin_vec![P(Ty::generic(
                                "List".to_string(),
                                thin_vec![P(Ty::simple("T".to_string()))]
                            ))]
                        )),]
                    ))),
                    None,
                )))),
            ],
        )
    ))),]));
    assert_eq!(result.unwrap(), expected);
}

#[test]
fn test_parse_generic_type_annotation() {
    let source = "let x: Array[int] = [1, 2, 3]";
    let tokens = Lexer::tokenize(source).unwrap();
    let mut parser = create_parser(tokens);
    let result = parser.parse_declaration();
    assert!(
        result.is_ok(),
        "Failed to parse generic type annotation: {}",
        result.unwrap_err()
    );
    let expected = P(AstElem::item(P(Item::bind(
        "x".to_string(),
        thin_vec![BindAttr {
            mutability: Mutability::Not,
            ref_kind: None
        }],
        Some(P(Ty::generic(
            "Array".to_string(),
            thin_vec![P(Ty::simple("int".to_string()))],
        ))),
        Some(P(Expr::array(thin_vec![
            P(Expr::literal(Literal::Int(1))),
            P(Expr::literal(Literal::Int(2))),
            P(Expr::literal(Literal::Int(3))),
        ]))),
    ))));
    assert_eq!(result.unwrap(), expected);
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
    let expected = P(AstElem::item(P(Item::bind(
        "x".to_string(),
        thin_vec![BindAttr {
            mutability: Mutability::Not,
            ref_kind: None
        }],
        Some(P(Ty::generic(
            "Map".to_string(),
            thin_vec![
                P(Ty::simple("String".to_string())),
                P(Ty::generic(
                    "Array".to_string(),
                    thin_vec![P(Ty::simple("int".to_string()))]
                ))
            ],
        ))),
        None,
    ))));
    assert_eq!(result, expected);
}

#[test]
fn test_parse_function_with_generic_return_type() {
    let source = "fn getValues() -> Array[int] { return [1, 2, 3] }";
    let tokens = Lexer::tokenize(source).unwrap();
    let mut parser = create_parser(tokens);
    let result = parser.parse_declaration();
    assert!(
        result.is_ok(),
        "Failed to parse function with generic return type: {}",
        result.unwrap_err()
    );
    let expected = P(AstElem::item(P(Item::fn_(
        "getValues".to_string(),
        thin_vec![],
        Function {
            generic_params: thin_vec![],
            inputs: thin_vec![],
            output: FnRetTy::Ty(P(Ty::generic(
                "Array".to_string(),
                thin_vec![P(Ty::simple("int".to_string()))],
            ))),
        },
        thin_vec![P(Statement::expr(P(Expr::return_(Some(P(Expr::array(
            thin_vec![
                P(Expr::literal(Literal::Int(1))),
                P(Expr::literal(Literal::Int(2))),
                P(Expr::literal(Literal::Int(3))),
            ]
        ))),))))],
    ))));
    assert_eq!(result.unwrap(), expected);
}
#[test]
fn test_parse_error_handling() {
    let tokens = vec![
        Token::Let,
        Token::Identifier("x".to_string()),
        Token::Assign,
        Token::IntLiteral(5),
        Token::Let, // This should cause an error because there's no Newline token before it
        Token::Identifier("y".to_string()),
        Token::Assign,
        Token::IntLiteral(10),
        Token::Newline,
        Token::Eof,
    ];
    let mut parser = create_parser(tokens);
    let result = parser.parse();
    assert!(result.is_err());
    match result {
        Err(ParserError::ExpectedToken(expected, found)) => {
            assert_eq!(expected, "newline");
            assert_eq!(found, "Let");
        }
        _ => panic!("Expected ParserError::ExpectedToken, got {:?}", result),
    }
}

#[test]
fn test_parse_multi_pipeline() {
    let input = r#"
        let processed = data
            |> map { x in x * 2 }
            |> filter { x in  x > 0 }
            |> fold(0) { acc, x in acc + x }
    "#;
    let tokens = Lexer::tokenize(input).unwrap();
    let mut parser = create_parser(tokens);
    let result = parser.parse();
    assert!(
        result.is_ok(),
        "Failed to parse pipeline: {}",
        result.unwrap_err()
    );
    let expected = P(AstElem::program(thin_vec![P(AstElem::expr(P(
        Expr::pipeline(
            P(AstElem::expr(P(Expr::pipeline(
                P(AstElem::expr(P(Expr::pipeline(
                    P(AstElem::item(P(Item::bind(
                        "processed".to_string(),
                        thin_vec![BindAttr {
                            mutability: Mutability::Not,
                            ref_kind: None
                        },],
                        None,
                        Some(P(Expr::path(None, Path::ident("data".to_string()))))
                    )))),
                    P(Expr::trailing_closure(
                        P(Expr::path(None, Path::ident("map".to_string()))),
                        thin_vec![P(Expr::path(None, Path::ident("x".to_string())))],
                        P(Expr::block(
                            thin_vec![P(Statement::expr(P(Expr::binary(
                                BinaryOperator::Multiply,
                                P(Expr::path(None, Path::ident("x".to_string()))),
                                P(Expr::literal(Literal::Int(2))),
                            ))))],
                            None
                        )),
                    )),
                )))),
                P(Expr::trailing_closure(
                    P(Expr::path(None, Path::ident("filter".to_string()))),
                    thin_vec![P(Expr::path(None, Path::ident("x".to_string())))],
                    P(Expr::block(
                        thin_vec![P(Statement::expr(P(Expr::binary(
                            BinaryOperator::GreaterThan,
                            P(Expr::path(None, Path::ident("x".to_string()))),
                            P(Expr::literal(Literal::Int(0))),
                        ))))],
                        None
                    )),
                )),
            )))),
            P(Expr::trailing_closure(
                P(Expr::call(
                    P(Expr::path(None, Path::ident("fold".to_string()))),
                    None,
                    thin_vec![P(Expr::literal(Literal::Int(0))),],
                )),
                thin_vec![
                    P(Expr::path(None, Path::ident("acc".to_string()))),
                    P(Expr::path(None, Path::ident("x".to_string()))),
                ],
                P(Expr::block(
                    thin_vec![P(Statement::expr(P(Expr::binary(
                        BinaryOperator::Add,
                        P(Expr::path(None, Path::ident("acc".to_string()))),
                        P(Expr::path(None, Path::ident("x".to_string()))),
                    ))))],
                    None
                )),
            )),
        )
    ))),]));
    assert_eq!(result.unwrap(), expected);
}

#[test]
fn test_simple_impl_block() {
    let input = r#"
        impl Display for Point {
            fn display(self) -> String {
                "Point({self.x}, {self.y})"
            }
        }
    "#;
    let tokens = Lexer::tokenize(input).unwrap();
    let mut parser = create_parser(tokens);
    let result = parser.parse();
    assert!(
        result.is_ok(),
        "Failed to parse impl block: {}",
        result.unwrap_err()
    );
    let expected = P(AstElem::program(thin_vec![P(AstElem::item(P(
        Item::impl_(
            "Display".to_string(),
            thin_vec![],
            ImplKind::Infer,
            "Point".to_string(),
            thin_vec![],
            None,
            thin_vec![],
            thin_vec![P(AstElem::item(P(Item::fn_(
                "display".to_string(),
                thin_vec![],
                Function {
                    generic_params: thin_vec![],
                    inputs: thin_vec![Param {
                        name: "self".to_string(),
                        ty: P(Ty::self_type()),
                    },],
                    output: FnRetTy::Ty(P(Ty::simple("String".to_string()))),
                },
                thin_vec![P(Statement::expr(P(Expr::literal(Literal::String(
                    "Point({self.x}, {self.y})".to_string()
                )))))],
            )))),],
        )
    ))),]));
    assert_eq!(result.unwrap(), expected);
}

#[test]
fn test_with_clause_simple() {
    let input = r#"
        fn generate_id() -> i32 with Random {
            next_int(0, 1000)
        }
        "#;
    let tokens = Lexer::tokenize(input).unwrap();
    let mut parser = Parser::new(tokens);
    let result = parser.parse();
    assert!(
        result.is_ok(),
        "Failed to parse with clause: {}",
        result.unwrap_err()
    );
    let expected = P(AstElem::program(thin_vec![P(AstElem::item(P(Item::fn_(
        "generate_id".to_string(),
        thin_vec![FnAttr::with_clause(thin_vec![P(WithClauseItem::Generic(
            GenericParam {
                name: "Random".to_string(),
                kind: GenericParamKind::Type(None),
                attrs: thin_vec![],
                bounds: None,
                is_placeholder: false,
            }
        )),])],
        Function {
            generic_params: thin_vec![],
            inputs: thin_vec![],
            output: FnRetTy::Ty(P(Ty::simple("i32".to_string()))),
        },
        thin_vec![P(Statement::expr(P(Expr::call(
            P(Expr::path(None, Path::ident("next_int".to_string()))),
            None,
            thin_vec![
                P(Expr::literal(Literal::Int(0))),
                P(Expr::literal(Literal::Int(1000))),
            ],
        ))))],
    )))),]));
    assert_eq!(result.unwrap(), expected);
}
