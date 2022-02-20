a = [2,3,4,6,10,11,14,16,18,20,34]
b = [1,2,5,7,8,9,10,15,18,29,40]
p = 0
q = 0
m = len(a)
n = len(b)
pre = a[0]
flag = 0
na = [a[0]]
nb = []
while p < m and q < n:
  if flag == 0:
    while  q < n and  b[q] <= pre:
      q += 1
    if q < n:
      pre = b[q]
      nb.append(b[q])
    flag = 1
    continue
  if flag == 1:
    while p < m and  a[p] <= pre:
      p += 1
    if p < m:
      pre = a[p]
      na.append(a[p])
    flag = 0
print(na,nb)