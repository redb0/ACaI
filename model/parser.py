from params import variables


def parse_obj_expr(obj_expr: str, delimiter=""):
    obj_expr = obj_expr.replace(' ', '')
    # tao = []
    tao_u = []
    tao_x = []
    a_names = []
    u_names = []
    x_names = []
    i = 0
    x_idx = -1
    u_idx = -1
    a_idx = -1
    model = ""
    obj = ""
    while i < (len(obj_expr)):
        # TODO: добавить еще помеху и h
        if (i + 1) != len(obj_expr) and (obj_expr[i + 1].isdigit() or obj_expr[i + 1] == "_" or obj_expr[i + 1] == "("):
            name = ""
            if obj_expr[i] == variables["object"]:
                name, t, x_idx, i = parse_var(i, obj_expr, x_idx, variables["object"], delimiter=delimiter)
                if name != "" and name not in x_names:
                    x_names.append(name)
                    tao_x.append(t)
            elif obj_expr[i] == variables["control"]:
                name, t, u_idx, i = parse_var(i, obj_expr, u_idx, variables["control"], delimiter=delimiter)
                if name != "" and name not in u_names:
                    u_names.append(name)
                    tao_u.append(t)
            elif obj_expr[i] == variables["coefficient"]:
                name, a_idx, i = parse_coefficient(i, obj_expr, a_idx, delimiter=delimiter)
                if name != "" and name not in a_names:
                    a_names.append(name)
            else:
                model += obj_expr[i]
                obj += obj_expr[i]
                i += 1
            if name != "":
                model += name
                obj += name
        else:
            model += obj_expr[i]
            obj += obj_expr[i]
            i += 1
    return model, obj, x_names, u_names, a_names, tao_x, tao_u


def is_operator(c: str) -> bool:
    operators = ["-", "+", "*", "/", "**"]
    if c in operators:
        return True
    return False


def is_int(number: str):
    try:
        int(number)
        return True
    except ValueError:
        print("Некорректный индекс вермени")
        return False


def parse_var(i: int, s: str, last_idx: int, variable, delimiter=""):
    tao = 0
    idx = -1
    # j = i
    if s[i] == variable:
        i += 1
        while i < len(s) and (not is_operator(s[i])) and (s[i] != ")"):
            if s[i] == delimiter or s[i] == "_":
                i += 1
            elif s[i].isdigit():
                idx, i = parse_index(i, s)
                if idx == -1:
                    idx = last_idx + 1
                if (s[i - 1] == "_") or (s[i - 1] == variable):
                    break
            elif s[i] == "(":
                i += 1
                if s[i] != variables["time"]:
                    print("Ожидается временной индекс")
                    return "", tao, idx, i
                while s[i] != ")":
                    if s[i] == variables["time"]:
                        i += 1
                    elif s[i] == "-":
                        i += 1
                        if not s[i].isdigit():
                            print("Ожидается так запаздывания")
                            return "", tao, idx, i
                    elif s[i].isdigit():
                        t = ""
                        while s[i].isdigit():
                            t += s[i]
                            i += 1
                        if is_int(t):
                            t = int(t)
                            if t > 0:
                                tao = t - 1
                        else:
                            print("Некорректный индекс вермени")
                            return "", tao, idx, i
                    else:
                        print("Недопустимый символ")
                        return "", tao, idx, i
                if i != len(s) and s[i] == ")":
                    i += 1
        if i == len(s) or s[i] == ")" or is_operator(s[i]):
            # if s.find(s[:j]) != -1:
            #     pass
            if idx != -1:
                name = variable + delimiter + str(idx)
            else:
                name = variable + delimiter + str(tao)
            return name, tao, idx, i
    else:
        print("Ожидается обозначение " + variable)


def parse_index(i: int, s: str):
    idx = ""
    if s[i].isdigit():
        # idx = s[i]
        # i += 1
        while i < len(s) and s[i].isdigit():
            idx += s[i]
            i += 1
        if is_int(idx) and int(idx) >= 0:
            return int(idx), i
        else:
            print("Некорректный индекс")
    else:
        print("Ожидается индекс")


def parse_coefficient(i: int, s: str, last_idx: int, delimiter=""):
    idx = -1
    if s[i] == variables["coefficient"]:
        i += 1
        while i < len(s) and (s[i].isdigit() or s[i] == "_"):
            if s[i] == delimiter or s[i] == "_":
                i += 1
            elif s[i].isdigit():
                idx, i = parse_index(i, s)
        if i == len(s) or is_operator(s[i]) or s[i] == ')':
            if idx == -1:
                idx = last_idx + 1
            name = variables["coefficient"] + delimiter + str(idx)
            return name, idx, i
    else:
        print("Ожидается обозначение " + variables["coefficient"])


def main():
    # parse_var_obj(0, "x(t-1)+a_0*u_0", 0, variable=variables["object"], delimiter="")
    # name, tao, idx, i = parse_var_obj(0, "x(t-1)+a_0*u_0", -1, "_")
    # parse_var_obj(7, "x(t-1)+a_0*u_0", 0, variable=variables["object"], delimiter="")
    # name, tao, idx, i = parse_var_obj(0, "x_5+a_0*u_0", -1, variable=variables["object"], delimiter="")
    # name, tao, idx, i = parse_var_obj(0, "x5", -1, variable=variables["object"], delimiter="")
    # name, tao, idx, i = parse_var_obj(0, "x(t-1)", -1, variable=variables["object"], delimiter="")
    # name, tao, idx, i = parse_var_obj(0, "x_2(t-3)", -1, variable=variables["object"], delimiter="")
    # name, tao, idx, i = parse_var(0, "u_2(t-3)", -1, variable=variables["control"], delimiter="")

    # model, tao = parse_obj_expr("x(t-1)+a_0*u_0", delimiter="")
    # model, obj, x_names, u_names, a_names, tao_x, tao_u = parse_obj_expr("a_0+a_1*x(t-1)+a_2*u(t-1)+a_3*sin(5*u(t-2))", delimiter="")
    # model, obj, x_names, u_names, a_names, tao = parse_obj_expr("a_0", delimiter="")

    # print(name)
    # print(tao)
    # print(idx)
    # print(i)

    model, obj, x_names, u_names, a_names, tao_x, tao_u = parse_obj_expr("a_0+a_1*x(t-1)+a_2*u(t-1)+a_3*sin(5*u(t-1))",
                                                                         delimiter="")

    print(model)
    print(obj)
    print(x_names)
    print(u_names)
    print(a_names)
    print(tao_x)
    print(tao_u)

if __name__ == "__main__":
    main()
