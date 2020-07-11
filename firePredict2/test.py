with open('testest.txt', 'r') as file:
    file2 = open("./save.csv", 'w+')

    lines = file.readlines() + [' ']

    for line in lines:
        print(line.strip('\n'))

        line = line.strip()
        file2.write(line)