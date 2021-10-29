from Pyfhel import Pyfhel, PyPtxt, PyCtxt

if __name__ == '__main__':
    HE = Pyfhel()           # Creating empty Pyfhel object
    HE.contextGen(p=65537)  # Generating context. The p defines the plaintext modulo.
                            #  There are many configurable parameters on this step
                            #  More info in Demo_ContextParameters, and
                            #  in Pyfhel.contextGen()
    HE.keyGen()             # Key Generation: generates a pair of public/secret keys
    print("2. Context and key setup")
    print(HE)
    integer1 = 127
    integer2 = -2
    ctxt1 = HE.encryptInt(integer1)  # Encryption makes use of the public key
    ctxt2 = HE.encryptInt(integer2)  # For integers, encryptInt function is used.
    print("3. Integer Encryption")
    print("    int ", integer1, '-> ctxt1 ', type(ctxt1))
    print("    int ", integer2, '-> ctxt2 ', type(ctxt2))
    print(ctxt1)
    print(ctxt2)
    ctxtSum = ctxt1 + ctxt2  # `ctxt1 += ctxt2` for quicker inplace operation
    ctxtSub = ctxt1 - ctxt2  # `ctxt1 -= ctxt2` for quicker inplace operation
    ctxtMul = ctxt1 * ctxt2  # `ctxt1 *= ctxt2` for quicker inplace operation
    print("4. Operating with encrypted integers")
    print(f"Sum: {ctxtSum}")
    print(f"Sub: {ctxtSub}")
    print(f"Mult:{ctxtMul}")
    resSum = HE.decryptInt(ctxtSum)  # Decryption must use the corresponding function
    #  decryptInt.
    resSub = HE.decryptInt(ctxtSub)
    resMul = HE.decryptInt(ctxtMul)
    print("#. Decrypting result:")
    print("     addition:       decrypt(ctxt1 + ctxt2) =  ", resSum)
    print("     substraction:   decrypt(ctxt1 - ctxt2) =  ", resSub)
    print("     multiplication: decrypt(ctxt1 + ctxt2) =  ", resMul)