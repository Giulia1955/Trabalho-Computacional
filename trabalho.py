import sympy
import math
import os
import traceback 

def parse_float(s: str) -> float:
    """Converte string para float, lidando com espaços."""
    s = s.strip()
    if not s:
        raise ValueError(f"String vazia após strip: '{s}'")
    try:
        return float(s)
    except ValueError:
        raise ValueError(f"Não foi possível converter '{s}' para float.")

def create_function(expression_string: str):
    expression_string = expression_string.strip()
    x = sympy.symbols('x')
    standard_expression = expression_string.replace('^', '**')
    
    local_namespace = {
        'e': sympy.E,
        'pi': sympy.pi,
        'exp': sympy.exp,
        'log': sympy.log,
        'log10': lambda x: sympy.log(x, 10),
        'sin': sympy.sin,
        'cos': sympy.cos,
        'tan': sympy.tan,
        'sqrt': sympy.sqrt
    }
    
    try:
        symbolic_expr = sympy.sympify(standard_expression, locals=local_namespace)
        f_callable = sympy.lambdify(x, symbolic_expr, modules='math')
        return f_callable
    except Exception as e:
        raise ValueError(f"Erro ao criar função '{expression_string}': {e}")

def bisseccao(f, a, b, tol=1e-2, max_iter=50):
    if f(a) * f(b) >= 0:
        return None, [("Intervalo inválido: f(a)*f(b) >= 0", "N/A", "N/A", "N/A", "N/A", "N/A")], 0, float('inf')
    
    iteracoes = []
    for i in range(1, max_iter + 1):
        c = (a + b) / 2
        fc = f(c)
        erro_estimado = (b - a) / 2
        iteracoes.append((i, round(a, 10), round(b, 10), round(c, 10), round(fc, 10), round(erro_estimado, 10)))
        
        if abs(erro_estimado) < tol:
            return c, iteracoes, i, abs(fc)
        
        if f(a) * fc < 0:
            b = c
        else:
            a = c
    
    c_final = (a + b) / 2
    fc_final = f(c_final)
    erro_final = (b - a) / 2
    error_row = ("Máx. iterações atingidas", round(a, 10), round(b, 10), round(c_final, 10), round(fc_final, 10), round(erro_final, 10))
    return None, iteracoes + [error_row], max_iter, abs(fc_final)

def newton(f, fprime, x0, tol=1e-2, round_digits: int = 4, max_iter=50):
    x = x0
    iteracoes = []
    for i in range(1, max_iter + 1):
        fx = round(f(x), round_digits)
        dfx = round(fprime(x), round_digits)
        
        if abs(dfx) < 1e-10:
            error_row = (i, round(x, round_digits), round(fx, round_digits), round(dfx, round_digits), None, "Derivada muito pequena")
            return None, iteracoes + [error_row], i, abs(fx)
        
        x_new = round(x - fx / dfx, round_digits)
        fx_new = round(f(x_new), round_digits)
        iteracoes.append((i, round(x, round_digits), round(fx, round_digits), round(dfx, round_digits), round(x_new, round_digits), round(fx_new, round_digits)))
        
        if abs(x_new - x) < tol or abs(fx_new) < tol:
            return x_new, iteracoes, i, abs(fx_new)
        
        x = x_new
    
    error_row = ("Máx. iterações atingidas", round(x, round_digits), round(f(x), round_digits), round(fprime(x), round_digits), None, None)
    return None, iteracoes + [error_row], max_iter, abs(f(x))

def MIL(f, itfunc, x0, tol=1e-2, round_digits = 4, max_iter=50):
    x = x0
    iteracoes = []
    for i in range (1, max_iter + 1):
        fx = round(f(x), round_digits)
        itfx = round(itfunc(x), round_digits)

        x_new = itfx
        fx_new = round(f(x_new), round_digits)
        iteracoes.append((i, round(x, round_digits), round(fx, round_digits), round(itfx, round_digits), round(x_new, round_digits), round(fx_new, round_digits)))

        if abs(x_new - x) < tol or abs(fx_new) < tol:
            return x_new, iteracoes, i, abs(fx_new)
        
        x = x_new

    error_row = ("Máx. iterações atingidas", round(x0, round_digits), round(f(x), round_digits), round(itfunc(x0), round_digits), None, None)
    return None, iteracoes + [error_row], max_iter, abs(f(x))

def secante (f, a, b, tol = 1e-2, round_digits: int = 4, max_iter = 50):
    x0 = a
    x1 = b
    iteracoes = []
    for i in range (1, max_iter + 1):
        fx0 = round(f(x0), round_digits)
        fx1 = round(f(x1), round_digits)
        x_new = round(x1 - (fx1 * (x1 - x0)) / (fx1 - fx0), round_digits)
        fx_new = round(f(x_new), round_digits)
        iteracoes.append((i, round(x0, round_digits), round(fx0, round_digits), round(fx1, round_digits), round(x_new, round_digits), round(fx_new, round_digits)))

        if abs(x_new - x1) < tol or abs(fx_new) < tol:
            return x_new, iteracoes, i, abs(fx_new)
        
        x0 = x1
        x1 = x_new
    error_row = ("Máx. iterações atingidas", round(x0, round_digits), round(f(x0), round_digits), round(fx_new, round_digits), None, None)
    return None, iteracoes + [error_row], max_iter, abs(f(x1))

def regula_falsi(f, a, b, tol=1e-2, round_digits: int = 4, max_iter=50):
    iteracoes = []
    fa = round(f(a), round_digits)
    fb = round(f(b), round_digits)

    if fa * fb > 0:
        raise ValueError("f(a) e f(b) devem ter sinais opostos para aplicar o método da Regula Falsi.")

    for i in range(1, max_iter + 1):
        x_new = round((a * fb - b * fa) / (fb - fa), round_digits)
        fx_new = round(f(x_new), round_digits)

        iteracoes.append((i, round(a, round_digits), round(fa, round_digits), round(fb, round_digits), round(x_new, round_digits), round(fx_new, round_digits)))

        if abs(fx_new) < tol or abs(b - a) < tol:
            return x_new, iteracoes, i, abs(fx_new)

        if fa * fx_new < 0:
            b, fb = x_new, fx_new
        else:
            a, fa = x_new, fx_new

    error_row = ("Máx. iterações atingidas", round(a, round_digits), round(fa, round_digits), round(fb, round_digits), None, None)
    return None, iteracoes + [error_row], max_iter, abs(fx_new)

def format_table(headers, rows, widths=None):
    if not rows:
        return "Nenhuma iteração realizada."
    
    rows_formatted = []
    for row in rows:
        if isinstance(row, str):
            rows_formatted.append(row)
        else:
            formatted_row = []
            for cell in row:
                if cell is None:
                    formatted_row.append("N/A")
                else:
                    formatted_row.append(str(cell))
            rows_formatted.append(tuple(formatted_row))
    
    rows = rows_formatted
    
    if widths is None:
        widths = [max(len(str(cell)) for cell in col) for col in zip(*rows)]
        widths = [max(w, len(h)) for w, h in zip(widths, headers)]
    
    header_line = " | ".join(f"{h:<{w}}" for h, w in zip(headers, widths))
    separator = "-+-".join("-" * w for w in widths)
    
    # Rows
    body_lines = []
    for row in rows:
        if isinstance(row, str):
            body_lines.append(row)
        else:
            body_lines.append(" | ".join(f"{cell:<{w}}" for cell, w in zip(row, widths)))
    
    return f"{header_line}\n{separator}\n" + "\n".join(body_lines)

# ---------------- PROCESSAMENTO ----------------

def process_functions(input_file: str, output_file: str):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Arquivo '{input_file}' não encontrado. Crie-o com expressões matemáticas, uma por linha.")
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        functions = [line.strip() for line in infile if line.strip()]
    
    if not functions:
        print("Aviso: Arquivo vazio ou sem funções válidas.")
        return
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for line_num, line in enumerate(functions, 1):
            try:
                # Split e strip para robustez
                parts = [part.strip() for part in line.split(",")]
                if len(parts) != 8:
                    raise ValueError(f"Linha inválida: esperados 7 campos, mas encontrou {len(parts)}")
                
                func_str = parts[0]
                itfunc_str = parts[1]
                dfunc_str = parts[2]
                
                # Parse numéricos com tratamento de erros
                a = parse_float(parts[3])
                b = parse_float(parts[4])
                x0 = parse_float(parts[5])
                tol = parse_float(parts[6])
                digits = int(parts[7])
                
                print(f"Processando: func='{func_str}', a={a}, b={b}, x0={x0}, tol={tol}")
                
                func = create_function(func_str)
                dfunc = create_function(dfunc_str)
                itfunc = create_function(itfunc_str)  
                
                f.write(f"\n{'='*60}\n")
                f.write(f"Função: {func_str}\n")
                f.write(f"{'='*60}\n")
                
                print("Função criada com sucesso.")
                
                # Bissecção
                raiz_bis, iters_bis, iters_count_bis, erro_bis = bisseccao(func, a, b, tol)
                headers_bis = ["Iter", "a", "b", "c", "f(c)", "Erro Estimado"]
                f.write(f"\n=== Método da Bissecção (a={a}, b={b}, tol={tol}) ===\n")
                f.write(format_table(headers_bis, iters_bis))
                print("Bissecção processada.")

                # Newton
                raiz_new, iters_new, iters_count_new, erro_new = newton(func, dfunc, x0, tol, digits)
                headers_new = ["Iter", "x", "f(x)", "f'(x)", "x_new", "f(x_new)"]
                f.write(f"\n\n=== Método de Newton (x0={x0}, tol={tol}) ===\n")
                f.write(format_table(headers_new, iters_new))
                print("Newton processado.")

                #MIL
                raiz_mil, iters_mil, iters_count_mil, erro_mil = MIL(func, itfunc, x0, tol, digits)
                headers_mil = ["Iter", "x", "f(x)", "itf(x)", "x_new", "f(x_new)"]
                f.write(f"\n\n=== Método de MIL (x0={x0}, tol={tol}) ===\n")
                f.write(format_table(headers_mil, iters_mil))
                print("MIL processado.")

                #Secante
                raiz_sec, iters_sec, iters_count_sec, erro_sec = secante(func, a, b, tol, digits)
                headers_sec = ["Iter", "x", "f(x)", "itf(x)", "x_new", "f(x_new)"]
                f.write(f"\n\n=== Método de Secante (x0={x0}, tol={tol}) ===\n")
                f.write(format_table(headers_sec, iters_sec))
                print("Secante processada.")

                #Regula_falsi
                raiz_reg, iters_reg, iters_count_reg, erro_reg = regula_falsi(func, a, b, tol, digits)
                headers_reg = ["Iter", "x", "f(x)", "itf(x)", "x_new", "f(x_new)"]
                f.write(f"\n\n=== Método Regula_falsi (x0={x0}, tol={tol}) ===\n")
                f.write(format_table(headers_reg, iters_reg))
                print("Regula Falsi processado.")
                
                # Tabela de Comparação
                f.write("\n\n=== Comparação dos Métodos ===\n")
                status_bis = "Convergiu" if raiz_bis is not None else "Falhou"
                status_new = "Convergiu" if raiz_new is not None else "Falhou"
                status_mil = "Convergiu" if raiz_mil is not None else "Falhou"
                status_sec = "Convergiu" if raiz_sec is not None else "Falhou"
                status_reg = "Convergiu" if raiz_reg is not None else "Falhou"
                headers_comp = ["Método", "Raiz Aproximada", "Iterações", "Erro Final |f(raiz)|", "Status"]
                comp_rows = [
                    ("Bissecção", f"{raiz_bis:.10f}" if raiz_bis is not None else "N/A", iters_count_bis, f"{erro_bis:.2e}", status_bis),
                    ("Newton", f"{raiz_new:.10f}" if raiz_new is not None else "N/A", iters_count_new, f"{erro_new:.2e}", status_new),
                    ("MIL", f"{raiz_mil:.10f}" if raiz_mil is not None else "N/A", iters_count_mil, f"{erro_mil:.2e}", status_mil),
                    ("Secante", f"{raiz_sec:.10f}" if raiz_sec is not None else "N/A", iters_count_sec, f"{erro_sec:.2e}", status_sec),
                    ("Regula_falsi", f"{raiz_reg:.10f}" if raiz_reg is not None else "N/A", iters_count_reg, f"{erro_reg:.2e}", status_reg)
                ]
                f.write(format_table(headers_comp, comp_rows))
                
                f.write(f"\n{'-'*60}\n")
                print(f"Processada com sucesso: {func_str}")
                
            except Exception as e:
                error_msg = f"Linha {line_num}: Erro ao processar '{line}': {e}"
                f.write(f"\n{error_msg}\n")
                print(error_msg)
                traceback.print_exc()  # Para debug no console

# Exemplo
if __name__ == "__main__":
    try:
        process_functions("funcoes.txt", "resultados.txt")
        print("\nProcessamento concluído. Verifique 'resultados.txt'.")
    except Exception as e:
        print(f"Erro geral: {e}")
        traceback.print_exc()
