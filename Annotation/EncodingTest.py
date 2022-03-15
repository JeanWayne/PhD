def tell_me_about(s): return (type(s), s)
w="Ãģ"
s='Á'

print(tell_me_about(s))

encod=s.encode('utf-8')
#uft8=encod.decode('uft-8')
print(encod.decode('latin-1'))
print(encod.decode('utf-8'))
print(encod.decode('iso8859-1'))
print(encod.decode('latin-1'))
print(encod.decode('ascii',errors="ignore"))
#udata=s.decode("utf-8")
#data=udata.encode("latin-1","ignore")

print(w.encode("iso-8859-1").decode("utf-8"))
#print(s.encode('uft-8'))
