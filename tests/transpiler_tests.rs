use alloy::lexer::Lexer;
use alloy::parser::Parser;
use alloy::transpiler::transpiler::transpile;

fn transpile_code(code: &str) -> String {
    let tokens = Lexer::tokenize(code).expect("Failed to tokenize code");
    let mut parser = Parser::new(tokens);
    let ast = parser.parse().expect("Failed to parse code");
    transpile(&ast)
}

#[test]
fn test_variable_declaration() {
    let alloy_code = "let x: int = 5;";
    let rust_code = transpile_code(alloy_code);
    assert_eq!(rust_code.trim(), "let x: i32 = 5;");
}

#[test]
fn test_mutable_variable_declaration() {
    let alloy_code = "let mut y: float = 3.14;";
    let rust_code = transpile_code(alloy_code);
    assert_eq!(rust_code.trim(), "let mut y: f64 = 3.14;");
}

#[test]
fn test_function_declaration() {
    let alloy_code = "func add(a: int, b: int) -> int { return a + b; }";
    let rust_code = transpile_code(alloy_code);
    assert_eq!(
        rust_code.trim(),
        "fn add(a: i32, b: i32) -> i32 {\n    return (a + b);\n}"
    );
}

#[test]
fn test_if_statement() {
    let alloy_code = "if (x > 5) { return true; } else { return false; }";
    let rust_code = transpile_code(alloy_code);
    assert_eq!(
        rust_code.trim(),
        "if (x > 5) {\n    return true;\n} else {\n    return false;\n}"
    );
}

#[test]
fn test_while_loop() {
    let alloy_code = "while (i < 10) { i = i + 1; }";
    let rust_code = transpile_code(alloy_code);
    assert_eq!(rust_code.trim(), "while (i < 10) {\n    i = (i + 1);\n}");
}

#[test]
fn test_for_loop() {
    let alloy_code = "for (let i = 0; i < 10; i = i + 1) { print(i); }";
    let rust_code = transpile_code(alloy_code);
    assert_eq!(
        rust_code.trim(),
        "for let i = 0; (i < 10); i = (i + 1) {\n    print(i);\n}"
    );
}

#[test]
fn test_array_literal() {
    let alloy_code = "let arr: [int] = [1, 2, 3, 4, 5];";
    let rust_code = transpile_code(alloy_code);
    assert_eq!(rust_code.trim(), "let arr: Vec<i32> = vec![1, 2, 3, 4, 5];");
}

#[test]
fn test_complex_expression() {
    let alloy_code = "let result = (a + b) * (c - d) / 2;";
    let rust_code = transpile_code(alloy_code);
    assert_eq!(rust_code.trim(), "let result = (((a + b) * (c - d)) / 2);");
}

#[test]
fn test_nested_function_calls() {
    let alloy_code = "let result = max(min(a, b), c);";
    let rust_code = transpile_code(alloy_code);
    assert_eq!(rust_code.trim(), "let result = max(min(a, b), c);");
}

#[test]
fn test_string_concatenation() {
    let alloy_code = r#"let full_name = first_name + " " + last_name;"#;
    let rust_code = transpile_code(alloy_code);
    assert_eq!(
        rust_code.trim(),
        r#"let full_name = ((first_name + " ") + last_name);"#
    );
}
