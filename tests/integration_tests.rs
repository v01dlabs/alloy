#![feature(box_patterns)]

use alloy::ast::AstNode;
use alloy::parser::Parser;

use alloy::type_checker::typecheck;
use alloy::Lexer;

#[test]
fn test_parse_complex_program() {
    let source = r#"
        fn processData[T](data: Array[T], predicate: |T| -> bool) -> int {
            let mut sum = 0
            for value in data {
                if predicate(value) {
                    sum = sum + 1
                }
            }
            return sum
        }

        fn main() {
            let numbers = [1, 2, 3, 4, 5]
            let result = processData[int](numbers) { value in
                value % 2 == 0
            }
            print("Even numbers count: " + result.toString())
        }
    "#;
    let tokens = Lexer::tokenize(source).unwrap();
    let mut parser = Parser::new(tokens);
    let result = parser.parse();

    assert!(
        result.is_ok(),
        "Failed to parse complex program: {}",
        result.unwrap_err()
    );
    if let Ok(box AstNode::Program(declarations)) = result {
        assert_eq!(
            declarations.len(),
            2,
            "Expected 2 declarations, got {}",
            declarations.len()
        );
        // Add other assertions later
    } else {
        panic!("Expected Program AST node");
    }
}

#[test]
fn test_typecheck_complex_program() {
    let code = r#"
            fn fibonacci(n: int) -> int {
                if (n <= 1) {
                    return n;
                } else {
                    return fibonacci(n - 1) + fibonacci(n - 2);
                }
            }

            fn main() -> int {
                let result: int = fibonacci(10);
                let numbers: [int] = [1, 2, 3, 4, 5];
                for (let i = 0; i < 5; i = i + 1) {
                    result = result + numbers[i];
                }
                return result;
            }
        "#;
    let tokens = Lexer::tokenize(code).unwrap();
    let mut parser = Parser::new(tokens);
    let ast = parser.parse().unwrap();
    assert!(typecheck(&ast).is_ok());
}
