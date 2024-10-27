use alloy::type_checker::typecheck;
use alloy::error::TypeError;
use alloy::Lexer;
use alloy::Parser;


fn type_check_code(code: &str) -> Result<(), TypeError> {
    let _ = tracing_subscriber::fmt::try_init();
    let tokens = Lexer::tokenize(code).map_err(|e| TypeError {
        message: e.to_string(),
    })?;
    println!("{:?}", tokens);
    let mut parser = Parser::new(tokens);
    let ast = parser.parse().map_err(|e| TypeError {
        message: e.to_string(),
    })?;
    println!("{:?}", ast);
    let res = typecheck(&ast);
    println!("{:?}", res);
    res
}

#[test]
fn test_typecheck_variable_declaration() {
    assert!(type_check_code("let x: int = 5;").is_ok());
    assert!(type_check_code("let x = 5;").is_ok());
    assert!(type_check_code("let x: float = 5.0;").is_ok());
    assert!(type_check_code("let x: int = 5.0;").is_err());
}

#[test]
fn test_typecheck_function_declaration() {
    assert!(type_check_code("fn add(a: int, b: int) -> int { return a + b; }").is_ok());
    assert!(type_check_code("fn add(a: int, b: int) -> float { return a + b; }").is_err());
}

#[test]
fn test_typecheck_if_statement() {
    assert!(type_check_code(
        "fn test(x: int) -> bool { if (x > 0) { return true; } else { return false; } }"
    )
    .is_ok());
    assert!(type_check_code(
        "fn test(x: int) -> bool { if (x) { return true; } else { return false; } }"
    )
    .is_err());
}

#[test]
fn test_typecheck_while_loop() {
    assert!(
        type_check_code("fn count_to_ten() { let i = 0; while (i < 10) { i = i + 1; } }").is_ok()
    );
    assert!(type_check_code("fn invalid() { while (1) { } }").is_err());
}

// stubbed out for new syntax
// #[test]
// fn test_typecheck_for_loop() {
//     assert!(type_check_code(
//         "fn count_to_ten() { for (let i = 0; i < 10; i = i + 1) { print(i); } }"
//     )
//     .is_ok());
//     assert!(type_check_code("func invalid() { for (let i = 0; i; i = i + 1) { } }").is_err());
// }

#[test]
fn test_typecheck_array_literal() {
    assert!(type_check_code("let arr: Array[int] = [1, 2, 3, 4, 5];").is_ok());
    assert!(type_check_code("let arr: Array[int] = [1, 2, 3, 4, true];").is_err());
}

#[test]
fn test_typecheck_arithmetic_operations() {
    assert!(type_check_code("fn add(a: int, b: int) -> int { return a + b; }").is_ok());
    assert!(type_check_code("fn add(a: int, b: float) -> float { return a + b; }").is_err());
}

#[test]
fn test_typecheck_boolean_operations() {
    assert!(type_check_code("fn and(a: bool, b: bool) -> bool { return a && b; }").is_ok());
    assert!(type_check_code("fn invalid(a: bool, b: int) -> bool { return a && b; }").is_err());
}

#[test]
fn test_typecheck_function_call() {
    let code = r#"
            fn add(a: int, b: int) -> int { return a + b; }
            fn main() -> int { return add(5, 3); }
        "#;
    assert!(type_check_code(code).is_ok());

    let invalid_code = r#"
            fn add(a: int, b: int) -> int { return a + b; }
            fn main() -> int { return add(5, "3"); }
        "#;
    assert!(type_check_code(invalid_code).is_err());
}
