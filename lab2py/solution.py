import sys

def parse_clause(clause_str):
    return set(clause_str.strip().split(' v '))

def is_tautology(clause):
    return any(negate(lit) in clause for lit in clause)

def load_resolution_input(file_path):
    with open(file_path, 'r') as f:
        lines = [line.strip().lower() for line in f.readlines() if line.strip()]

    clauses = [parse_clause(line) for line in lines[:-1]]
    goal = parse_clause(lines[-1])

    return clauses, goal, lines

def load_cooking_input(file_path):
    with open(file_path, 'r') as f:
        lines = [line.strip().lower() for line in f.readlines() if line.strip()]

    clauses = [parse_clause(line) for line in lines]

    return clauses, lines

def load_cooking_file(file_path):
    clauses = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip().lower()

            if not line:
                continue

            if line.endswith(('+', '-', '?')):
                sign = line[-1]
                clause_str = line[:-1].strip()
                parsed = parse_clause(clause_str)
                clauses.append((clause_str, parsed, sign))

    return clauses

def negate(literal):
    return literal[1:] if literal.startswith("~") else "~" + literal

def resolve(clause1, clause2):
    resolvents = []
    seen = set()

    for literal in clause1:
        negated = negate(literal)

        if negated in clause2:
            new_clause = (clause1 | clause2) - {literal, negated}

            if any(negate(lit) in new_clause for lit in new_clause):
                continue

            key = tuple(sorted(new_clause))

            if key not in seen:
                seen.add(key)
                resolvents.append(new_clause)

    return resolvents

def resolution(clauses, goal):
    clauses = [c for c in clauses if not is_tautology(c)]

    negated_goal_clauses = [{negate(lit)} for lit in goal]
    clauses += negated_goal_clauses

    seen = set(tuple(sorted(c)) for c in clauses)
    new = []
    while True:
        for i in range(len(clauses)):
            for j in range(i + 1, len(clauses)):
                for resolvent in resolve(clauses[i], clauses[j]):
                    key = tuple(sorted(resolvent))
                    if key not in seen:
                        if not resolvent:
                            return True
                        seen.add(key)
                        new.append(resolvent)

        if not new:
            return False

        clauses += new
        new = []

def cooking_resolution(clauses, cooking, lines):
    print("Constructed with knowledge:")

    for line in lines:
        print(line)

    print("\n")

    for entry in cooking:
        clause_full, clause, sign = entry

        print(f"User's command: {clause_full}")

        if sign == '?':
            if resolution(clauses, clause):
                print(f"[CONCLUSION]: {clause_full} is true")
            else:
                print(f"[CONCLUSION]: {clause_full} is unknown")

        elif sign == '+':
            if clause not in clauses:
                clauses.append(clause)
                print(f"added {clause_full}")

        elif sign == '-':
            if clause in clauses:
                clauses.remove(clause)
                print(f"removed {clause_full}")

        print("\n")

def main():
    if len(sys.argv) not in [3, 4]:
        print("Usage: python solution.py [resolution | cooking] <input_file> [<cooking_file>]")
        sys.exit(1)

    mode = sys.argv[1]
    input_file = sys.argv[2]

    if mode == "resolution":
        if len(sys.argv) != 3:
            print("Usage: python solution.py resolution <input_file>")
            sys.exit(1)

        clauses, goal, lines = load_resolution_input(input_file)
        goal_line = lines[-1]

        result = resolution(clauses, goal)

        if result:
            print(f"[CONCLUSION]: {goal_line} is true")
        else:
            print(f"[CONCLUSION]: {goal_line} is unknown")

    elif mode == "cooking":
        if len(sys.argv) != 4:
            print("Usage: python solution.py cooking <input_file> <cooking_file>")
            sys.exit(1)

        cooking_file = sys.argv[3]

        clauses, lines = load_cooking_input(input_file)
        cooking = load_cooking_file(cooking_file)

        cooking_resolution(clauses, cooking, lines)

    else:
        print("Usage: python solution.py [resolution | cooking] <input_file> [<cooking_file>]")
        sys.exit(1)

if __name__ == "__main__":
    main()
