from rocketcea.cea_obj import CEA_Obj

cr = 6.25 # contraction ratio
ispObj = CEA_Obj( oxName='LOX', fuelName='CH4', fac_CR=cr)

# Use 100 atm to make output easy to read
Pc = 5 * 145.0377 # chamber pressure in psi

# use correlation to make 1st estimate of Pcinj_face / Pcomb_end
PinjOverPcomb = 1.0 + 0.54 / cr**2.2

# use RocketCEA to refine initial estimate
PinjOverPcomb = ispObj.get_Pinj_over_Pcomb( Pc=Pc * PinjOverPcomb, MR=3.0 )

# print results (noting that "COMB END" == 100.00 atm)
s = ispObj.get_full_cea_output( Pc=Pc * PinjOverPcomb, MR=3.0, eps=7.61)
print( s )

print(f"Isp:\t{ispObj.get_Isp( Pc=Pc * PinjOverPcomb, MR=3.0, eps=7.61)}")
print(f"Cstar:\t{ispObj.get_Cstar( Pc=Pc * PinjOverPcomb, MR=3.0)}")
print(f"enthalpy:\t{ispObj.get_Enthalpies( Pc=Pc * PinjOverPcomb, MR=3.0, eps=7.61, frozen=True )}")
print(f"entropy:\t{ispObj.get_Entropies( Pc=Pc * PinjOverPcomb, MR=3.0, eps=7.61, frozen=True )}")
print(f"density:\t{ispObj.get_Densities( Pc=Pc * PinjOverPcomb, MR=3.0, eps=7.61, frozen=True )}")
MolWt, Gamma = ispObj.get_Chamber_MolWt_gamma( Pc=Pc * PinjOverPcomb, MR=3.0)
print(f"MolWt_Gamma:\t{ispObj.get_Chamber_MolWt_gamma( Pc=Pc * PinjOverPcomb, MR=3.0 )}")
print(f"Tcomb:\t{ispObj.get_Tcomb(Pc=Pc * PinjOverPcomb, MR=3.0) / 1.8}")
print(f"T:\t{ispObj.get_Temperatures( Pc=Pc * PinjOverPcomb, MR=3.0, eps=7.61 )}")
print(f"Cp:\t{ispObj.get_Chamber_Cp( Pc=Pc * PinjOverPcomb, MR=3.0 )}, calc = {Gamma * 8314.46 / MolWt / (Gamma - 1)}")