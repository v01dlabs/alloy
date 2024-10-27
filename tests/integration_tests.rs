#![feature(box_patterns)]

use alloy::ast::AstNode;
use alloy::parser::Parser;

use alloy::type_checker::typecheck;
use alloy::Lexer;

fn init_tracing() {
    let format = tracing_subscriber::fmt::format()
        .pretty();

    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .event_format(format)
        .with_test_writer()
        .with_ansi(true)
        .try_init();
}

#[test]
fn test_parse_complex_program() {
    init_tracing();
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
    
}

#[test]
fn test_typecheck_complex_program() {
    init_tracing();
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
                let numbers: Array[int] = [1, 2, 3, 4, 5];
                for num in numbers {
                    result = result + num;
                }
                return result;
            }
        "#;
    let tokens = Lexer::tokenize(code).unwrap();
    let mut parser = Parser::new(tokens);
    let ast = parser.parse();
    assert!(
        ast.is_ok(),
        "Failed to parse complex program: {}",
        ast.unwrap_err()
    );
    let check_result = typecheck(&ast.unwrap());
    assert!(
        check_result.is_ok(),
        "Failed to typecheck complex program: {}",
        check_result.unwrap_err()
    );
}
