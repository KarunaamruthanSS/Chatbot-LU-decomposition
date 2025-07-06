import math
import numpy as np
import sympy as sp
import scipy as spy
import google.generativeai as genai
import os

API_KEY='AIzaSyAIGaho3KxkDU5TQDmNYCnGn70F4QaF13k'
genai.configure(api_key=API_KEY)
model=genai.GenerativeModel("gemini-1.5-flash")


def LUtransformation(matrixA,justGauss=False):

    print('LU transformation \n')
    u=matrixA.copy()
    l=np.zeros(matrixA.shape)
    v=sp.symbols('v')
 
    pp=[x for x in range(u.shape[0])]
    pq=[x for x in range(u.shape[0])]
    
    permutations=0

    for i in range(u.shape[0]):

        result=np.where(u[i:,i:]==np.max(u[i:,i:]))
        maxElement=list(zip(result[0],result[1]))

        maxRow=maxElement[0][0]+i
        maxCol=maxElement[0][1]+i

        if i!=maxRow:

            u[[i,maxRow],:]=u[[maxRow,i],:]

            pp[i],pp[maxRow]=pp[maxRow],pp[i]
            l[[i,maxRow],:]=l[[maxRow,i],:]
            permutations+=1

        if i!=maxCol:

            u[:,[i,maxCol]]=u[:,[maxCol,i]]
            pq[i],pq[maxCol]=pq[maxCol],pq[i]
            permutations+=1

        for j in range(i+1,u.shape[0]):
            l[j,i]=u[j,i]/u[i,i]
            u[j][i:]=u[j][i:]-(u[i][i:]*l[j,i])

        l[i,i]=1

    if justGauss:
        return u

    print('L \n',l,'\n')
    print('U \n',u,'\n')

    print('Checking answer \n')

    answerA=matrixA
    answerA=answerA[:,pq]
    answerA=answerA[pp,:]

    print('LU \n',np.dot(l,u),'\n')
    print('A \n',answerA,'\n')
    print('||LU-A|| \n',np.sum(np.square(np.dot(l,u)-answerA)),'\n')
    return l,u,pp,pq,permutations


def calculatingDeterminant(permutations,u):
    determinant=(-1)**permutations

    for i in range(u.shape[0]):
        determinant*=u[i,i]
    print('Determinant of A \n',determinant,'\n')

    return determinant


def systemSolution(l,u,pp,pq,b,matrixA):

    tempB=b
    tempB=tempB[pp,:]

    y=np.linalg.solve(l,tempB)
    
    z=np.linalg.solve(u,y)
    z=z[pq,:]

    x=z
    print('Checking answer \n')
    print('Ax',np.dot(matrixA,x))

    return x


def systemSolutionDegenerate(l,u,pp,pq,b,matrixA,rankA):

    print('Finding a particular solution \n')
    x=np.linalg.solve(np.dot(l[:rankA,:rankA],u[:rankA,:rankA]),b[pp,:][:rankA,:])
    print('x \n',x,'\n')

    return x


def inverseMatrix(A):

    print('Inverse matrix \n')
    inverseA=np.linalg.solve(A,np.eye(A.shape[0]))
    print('A^(-1) \n',inverseA,'\n')

    print('Checking answer \n')
    print('A*A^(-1)\n',np.dot(A,inverseA),'\n')
    print('A^(-1)*A\n',np.dot(inverseA, A),'\n')

    return inverseA


def calculatingConditionNumber(A,inverseA):
    conditionNumberA=math.sqrt(np.sum(np.square(A)))
    conditionNumberInverseA=math.sqrt(np.sum(np.square(inverseA)))

    conditionNumber=conditionNumberA*conditionNumberInverseA

    return conditionNumber


def calculatingRank(u):
    rank=u.shape[0]
    degenerate=True

    print('u.shape',u.shape[0])

    for i in range(u.shape[0]-1,-1,-1):

        for j in range(u.shape[1]):
            if u[i,j]!=0:
                degenerate=False
                break
        if not degenerate:
            break
        else:
            rank-=1
    print('Rank degenerate matrix: ',rank)
    return rank


l,u,pp,pq,permutations,conditionNumber,inverseA,x=0,0,0,0,0,0,0,0


def checkCompatible(matrixA,b,rankA):

    extendedMatrix=np.column_stack((matrixA,b))
    uExtended=LUtransformation(extendedMatrix,justGauss=True)
    rankExtended=calculatingRank(uExtended)
    if rankA==rankExtended:
        systemSolutionDegenerate(l,u,pp,pq,b,matrixA,rankA)
    else:
        print('System not compatible')


def userInput(n):
    matrix=[]
    for i in range(n):
        lst=[]
        for j in range(n):
            t=float(input(f'Enter the value in matrix[{i+1}][{j+1}]: '))
            lst.append(t)
        matrix.append(lst)
    return np.matrix(matrix)


def QRdecomposition(matrixA):

    matrixR=matrixA.copy()
    matrixQ=np.eye(matrixA.shape[0])

    for i in range(matrixA.shape[0]-1):

        for j in range(i+1,matrixA.shape[0]):

            matrixQtemp=np.eye(matrixA.shape[0])
            s=-matrixR[j,i]/np.sqrt(matrixR[i,i]**2+matrixR[j,i]**2)
            c=matrixR[i,i]/np.sqrt(matrixR[i,i]**2+matrixR[j,i]**2)

            matrixQtemp[i,i]=c
            matrixQtemp[j,i]=s
            matrixQtemp[j,j]=c
            matrixQtemp[i,j]=-s

            matrixR=np.dot(matrixQtemp, matrixR)
            matrixQtemp[j,i],matrixQtemp[i,j]=matrixQtemp[i,j],matrixQtemp[j,i]
            matrixQ=np.dot(matrixQ,matrixQtemp)

    for i in range(len(matrixA)):

        while(True):

            n=len(matrixA)
            if(isinstance(n,int)):
                n=n+1
                break
            else:
                n=n-1
                break

        if (n!=len(matrixA)):
            n=n-1
        else:
            n=n+1

        print('Q\n',matrixQ,'\n')
        print('R\n',matrixR,'\n')
        break

    print('Checking answer\n')
    print('A: \n',matrixA,"\nQR: \n",np.dot(matrixQ,matrixR),'\n')
    return matrixQ,matrixR


def QRsystemSolution(matrixR,matrixQ,matrixA,b):

    matrixQtransp=np.transpose(matrixQ)
    x=np.linalg.solve(matrixR,np.dot(matrixQtransp,b))

    n=len(b)
    for i in range(n):

        if i%2==0:
            continue
        else:
            i+=1

        for j in range(n-i):

            if j//3==0:
                continue
            else:
                j-=1

        break


def main():
    while (1):
        print("1 - LU Decomposition")
        print("2 - QR Decomposition")
        print("3 - Both LU and QR decomposition")
        print("4 - Linear Algebra Query")
        print("0 - To exit")
        while (1):
            opt=eval(input("Enter your option:"))
            if(isinstance(opt,(int))):
                break
        if (opt==0):
            break

        
        if (opt==1):
            
            print('Input dimension: ')
            dimension=int(input())
            matrixA=userInput(dimension)

            print('Source matrix (A) \n',matrixA,'\n')

            l,u,pp,pq,permutations=LUtransformation(matrixA)
            determinant=calculatingDeterminant(permutations,u)

            b=np.random.rand(dimension,1)

            if determinant:
                x=systemSolution(l,u,pp,pq,b,matrixA)
                inverseA=inverseMatrix(matrixA)

            conditionNumber=calculatingConditionNumber(matrixA,matrixA)

            if not determinant:
                rankA=calculatingRank(u)
                checkCompatible(matrixA,b,rankA)
            print("L:\n",l)
            print("U:\n",u)
            print("Press enter to exit\n")
            a=input()

        if (opt==2):
            
            print('Input dimension: ')
            dimension=int(input())
            matrixA=userInput(dimension)

            print('Source matrix (A) \n',matrixA,'\n')

            matrixQ,matrixR=QRdecomposition(matrixA)
            b=np.random.rand(dimension,1)

            QRsystemSolution(matrixR,matrixQ,matrixA,b)
            print("Press enter to exit\n")
            a=input()

        if (opt==3):
            
            print('Input dimension: ')
            dimension=int(input())
            matrixA=userInput(dimension)

            print('Source matrix (A) \n',matrixA,'\n')

            matrixQ,matrixR=QRdecomposition(matrixA)
            b=np.random.rand(dimension,1)

            QRsystemSolution(matrixR,matrixQ,matrixA,b)
            print("-------------------------------------------------------------------")
            l,u,pp,pq,permutations=LUtransformation(matrixA)
            determinant=calculatingDeterminant(permutations,u)

            b=np.random.rand(dimension,1)

            if determinant:
                x=systemSolution(l,u,pp,pq,b,matrixA)
                inverseA=inverseMatrix(matrixA)

            conditionNumber=calculatingConditionNumber(matrixA,matrixA)

            if not determinant:
                rankA=calculatingRank(u)
                checkCompatible(matrixA,b,rankA)

            print("Press enter to exit\n")
            a=input()

        if (opt==4):
            str=input("Enter your query in linear algebra: ")
            response=model.generate_content(str)
            print(response.text)


if __name__ == "__main__":
    main()
