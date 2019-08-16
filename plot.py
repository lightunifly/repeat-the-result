import matplotlib.pyplot as plt

plt.xlabel('iteration')
plt.ylabel('u(0,(100,100,...,100))')
plt.plot(ans)
plt.savefig('Black-Scholes equation ans.png')
plt.clf()
plt.xlabel('iteration')
plt.ylabel('~cost')
plt.plot(loss)
plt.savefig('Black-Scholes equation loss.png')