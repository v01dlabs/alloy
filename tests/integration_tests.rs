use alloy::lexer::Lexer;
use alloy::parser::{AstNode, Parser};

#[test]
fn test_parse_complex_program() {
    let source = r#"
        fn processData[T](data: Array[T], predicate: (T) -> bool) -> int {
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
        "Failed to parse complex program: {:?}",
        result
    );
    if let Ok(AstNode::Program(declarations)) = result {
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
