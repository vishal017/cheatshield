def log_violation(message):
    print(f"[VIOLATION] {message}")
    # Optionally, write to a log file
    with open("violations.log", "a") as f:
        f.write(f"{message}\n")