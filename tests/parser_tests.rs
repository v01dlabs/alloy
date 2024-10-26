#![feature(box_patterns)]

use alloy::{
    ast::{ty::{FnRetTy, Function, GenericParam, Ty, TyKind}, AstNode, BinaryOperator, Precedence},
    error::ParserError,
    lexer::{Lexer, Token},
    parser::{parse, Parser},
};
use thin_vec::thin_vec;

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
    let result = parser.parse_declaration();
    assert!(
        result.is_ok(),
        "Failed to parse variable declaration: {}",
        result.unwrap_err()
    );
    assert!(matches!(result,
        Ok(box AstNode::VariableDeclaration {
            name,
            attrs: _,
            type_annotation: Some(box Ty { kind: TyKind::Simple(type_name) }),
            initializer: Some(box AstNode::IntLiteral(5))
        }) if name == "x" && type_name == "int"
    ));
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

    match result {
        box AstNode::FunctionDeclaration {
            name,
            attrs: _,
            function:
                Function {
                    generic_params,
                    inputs: params,
                    output: return_type,
                },
            body,
        } => {
            assert_eq!(name, "add");
            assert_eq!(
                generic_params,
                thin_vec![GenericParam::simple("T".to_string())]
            );
            assert_eq!(params.len(), 2);
            assert_eq!(params[0].name, "a");
            assert_eq!(params[1].name, "b");
            assert!(matches!(&params[0].ty.kind, TyKind::Simple(t) if t == "T"));
            assert!(matches!(&params[1].ty.kind, TyKind::Simple(t) if t == "T"));
            assert!(
                matches!(return_type, FnRetTy::Ty(box Ty { kind: TyKind::Simple(t) }) if t == "T")
            );
            assert_eq!(body.len(), 1);
            match body[0] {
                box AstNode::ReturnStatement(Some(box AstNode::BinaryOperation { .. })) => {}
                _ => panic!("Expected return statement with binary operation"),
            }
        }
        _ => panic!("Expected FunctionDeclaration"),
    }
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
    let mut parser = Parser::new(tokens);
    let result = parser.parse();
    assert!(result.is_ok());
    if let Ok(box AstNode::Program(declarations)) = result {
        assert_eq!(declarations.len(), 2);
        assert!(matches!(declarations[0].clone(),
            box AstNode::VariableDeclaration { name, .. } if name == "x"));
        assert!(matches!(declarations[1].clone(),
            box AstNode::VariableDeclaration { name, .. } if name == "y"));
    } else {
        panic!("Expected Program with two declarations");
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
        box AstNode::IfStatement {
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
    let result = parser.parse_statement();
    assert!(
        result.is_ok(),
        "Failed to parse while loop: {}",
        result.unwrap_err()
    );
    if let Ok(box AstNode::WhileLoop { condition, body }) = result {
        assert!(matches!(*condition, AstNode::BinaryOperation { .. }));
        assert!(matches!(*body, AstNode::Block(statements) if statements.len() == 1));
    } else {
        panic!("Expected WhileLoop AST node");
    }
}

#[test]
fn test_parse_for_statement() {
    let source = "for name in names { print(name) }";
    let tokens = Lexer::tokenize(source).unwrap();
    let mut parser = Parser::new(tokens);
    let result = parser.parse_statement();
    assert!(
        result.is_ok(),
        "Failed to parse for statement: {}",
        result.unwrap_err()
    );
    if let Ok(box AstNode::ForInLoop {
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
    assert!(matches!(
        result,
        box AstNode::PipelineOperation {
            left: box AstNode::PipelineOperation { .. },
            right: box AstNode::FunctionCall { .. },
        }
    ));
}

#[test]
fn test_parse_trailing_closure() {
    let source = "someFunction() { in return 42 }";
    let tokens = Lexer::tokenize(source).unwrap();
    println!("{:?}", tokens);
    let mut parser = Parser::new(tokens);
    let result = parser.parse_expression(Precedence::None);
    assert!(
        result.is_ok(),
        "Failed to parse trailing closure: {}",
        result.unwrap_err()
    );
    if let Ok(box AstNode::TrailingClosure { callee, closure }) = result {
        assert!(matches!(*callee, AstNode::FunctionCall { .. }));
        assert!(matches!(*closure, AstNode::Block(..)));
    } else {
        panic!("Expected TrailingClosure, got {:?}", result);
    }
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
        box AstNode::GuardStatement {
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
    let result = parser.parse_statement();
    println!("{:?}", result);
    assert!(
        result.is_ok(),
        "Failed to parse return statement: {}",
        result.unwrap_err()
    );
    assert!(matches!(
        result,
        Ok(box AstNode::ReturnStatement(Some(box AstNode::BinaryOperation { .. })))
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
        box AstNode::BinaryOperation {
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
fn test_basic_effect() {
    let input = r#"
    effect IO {
        read_line() -> String
        print(str: String)
    }
    "#;
    let tokens = Lexer::tokenize(input).unwrap();
    let mut parser = Parser::new(tokens);
    let result = parser.parse();
    assert!(
        result.is_ok(),
        "Failed to parse effect declaration: {}",
        result.unwrap_err()
    );
    if let Ok(box AstNode::Program(declarations)) = result {
        assert_eq!(declarations.len(), 1);
        let declaration = declarations.first().unwrap();
        assert!(matches!(declaration,
            box AstNode::EffectDeclaration { .. } 
        ));
        if let box AstNode::EffectDeclaration { name, members, .. } = declaration {
            assert_eq!(name, "IO");
            assert_eq!(members.len(), 2);
            assert!(matches!(members[0].clone(), box AstNode::FunctionDeclaration { name, 
                function: Function { inputs, output, .. }, .. } if name == "read_line" && inputs.len() == 0 
                && matches!(output.clone(), FnRetTy::Ty(box Ty { kind: TyKind::Simple(type_name), .. }) if type_name == "String")));
            assert!(matches!(members[1].clone(), box AstNode::FunctionDeclaration { name, 
                function: Function { inputs, output, .. }, .. } if name == "print" && inputs.len() == 1 
                && matches!(output.clone(), FnRetTy::Ty(box Ty { kind: TyKind::Tuple(t)}) if t.len() == 0)));
        }
    } else {        
        panic!("Expected EffectDeclaration, got {:?}", result);
    }
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
    let mut parser = Parser::new(tokens);
    let result = parser.parse();
    assert!(
        result.is_ok(),
        "Failed to parse struct declaration: {}",
        result.unwrap_err()
    );
    if let Ok(box AstNode::Program(declarations)) = result {
        assert_eq!(declarations.len(), 1);
        let declaration = declarations.first().unwrap();
        assert!(matches!(declaration,
            box AstNode::StructDeclaration { .. } 
        ));
        if let box AstNode::StructDeclaration { name, members, .. } = declaration {
            assert_eq!(name, "List");
            assert_eq!(members.len(), 2);
            assert!(matches!(members[0].clone(), 
                box AstNode::VariableDeclaration { name, 
                    type_annotation: Some(box Ty { kind: TyKind::Generic(type_name,..), .. }), .. 
                } if name == "head" && type_name == "Option")
            );
            assert!(matches!(members[1].clone(), 
                box AstNode::VariableDeclaration { name, 
                    type_annotation: Some(box Ty { kind: TyKind::Generic(type_name, ..), .. }), .. 
                } if name == "tail" && type_name == "Option")
            );
            // TODO: verify the full generic type
        }
    }
}

#[test]
fn test_parse_generic_type_annotation() {
    let source = "let x: Array[int] = [1, 2, 3]";
    let tokens = Lexer::tokenize(source).unwrap();
    let mut parser = Parser::new(tokens);
    let result = parser.parse_declaration();
    assert!(
        result.is_ok(),
        "Failed to parse generic type annotation: {}",
        result.unwrap_err()
    );
    if let Ok(box AstNode::VariableDeclaration {
        name,
        attrs: _,
        type_annotation,
        initializer,
    }) = result
    {
        assert_eq!(name, "x");
        assert!(
            matches!(type_annotation, Some(box Ty { kind: TyKind::Generic(base_type, params) }) if base_type == "Array" && params.len() == 1)
        );
        assert!(matches!(initializer, Some(box AstNode::ArrayLiteral(..))));
    } else {
        panic!("Expected VariableDeclaration, got {}", result.unwrap_err());
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
        box AstNode::VariableDeclaration {
            name,
            attrs: _,
            type_annotation: Some(box Ty { kind: TyKind::Generic(base_type, params) }),
            initializer: None
        } if name == "x" && base_type == "Map" && params.len() == 2
    ));
}

#[test]
fn test_parse_function_with_generic_return_type() {
    let source = "fn getValues() -> Array[int] { return [1, 2, 3] }";
    let tokens = Lexer::tokenize(source).unwrap();
    let mut parser = Parser::new(tokens);
    let result = parser.parse_declaration();
    assert!(
        result.is_ok(),
        "Failed to parse function with generic return type: {}",
        result.unwrap_err()
    );
    if let Ok(box AstNode::FunctionDeclaration {
        name,
        attrs: _,
        function:
            Function {
                generic_params,
                inputs: params,
                output: return_type,
            },
        body,
    }) = result
    {
        assert_eq!(name, "getValues");
        assert!(generic_params.is_empty());
        assert!(params.is_empty());
        assert!(
            matches!(return_type, FnRetTy::Ty(box Ty { kind: TyKind::Generic(base_type, type_params) }) if base_type == "Array" && type_params.len() == 1)
        );
        assert!(!body.is_empty());
    } else {
        panic!("Expected FunctionDeclaration, got {:?}", result);
    }
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
    let mut parser = Parser::new(tokens);
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
fn test_parse_pipeline_basic() {
    let input = r#"
        let processed = data
            |> map(x -> x * 2)
            |> filter(x -> x > 0)
            |> fold(0, (acc, x) -> acc + x)
    "#;
    let tokens = Lexer::tokenize(input).unwrap();
    let mut parser = Parser::new(tokens);
    let result = parser.parse();
    assert!(
        result.is_ok(),
        "Failed to parse pipeline: {}",
        result.unwrap_err()
    );
    if let Ok(box AstNode::Program(declarations)) = result {
        assert_eq!(declarations.len(), 1);
        let declaration = declarations.first().unwrap();
        assert!(matches!(declaration,
            box AstNode::VariableDeclaration { name, initializer: Some(box AstNode::PipelineOperation { .. }), .. } 
            if name == "processed"
        ));
        if let box AstNode::VariableDeclaration { name, initializer: Some(box AstNode::PipelineOperation { left, right }), .. } = declaration {
            assert_eq!(name, "processed");
            assert!(matches!(left.clone(), box AstNode::FunctionCall { .. }));
            assert!(matches!(right.clone(), box AstNode::FunctionCall { .. }));
        }
    }
}