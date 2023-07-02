def levenshtein_distance(str1, str2):
    m = len(str1)
    n = len(str2)

    d = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j

    for j in range(1, n + 1):
        for i in range(1, m + 1):
            if str1[i - 1] == str2[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i][j - 1],      # Insert
                                  d[i - 1][j],      # Delete
                                  d[i - 1][j - 1])  # Replace
    return d[m][n]
