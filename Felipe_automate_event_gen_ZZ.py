from hmac import new
import re 
import os
import subprocess

# Path to MG5 installation - assumes only one installation
mg5_install_dir = "./MG5_aMC_v*"

# Generate process
subprocess.run(
    [os.path.join(mg5_install_dir, 'bin', 'mg5_aMC')],
    input='define p = 1 2 3 4 5 -1 -2 -3 -4 -5 21\ndefine j = p\ngenerate p p > e+ e- mu+ mu-\noutput Felipe_pp_ZZ_4l\nquit\n',
    text=True,
    check=True
)

process_dir = os.path.join(mg5_install_dir, "Felipe_pp_ZZ_4l")

# Edit run_card.dat
run_card_path = os.path.join(process_dir, "Cards", "run_card.dat")
new_run_card = """
#*********************************************************************
#                       MadGraph5_aMC@NLO                            *
#                                                                    *
#                     run_card.dat MadEvent                          *
#                                                                    *
#  This file is used to set the parameters of the run.               *
#                                                                    *
#  Some notation/conventions:                                        *
#                                                                    *
#   Lines starting with a '# ' are info or comments                  *
#                                                                    *
#   mind the format:   value    = variable     ! comment             *
#                                                                    *
#   To display more options, you can type the command:               *
#      update to_full                                                *
#*********************************************************************
#                                                                    
#*********************************************************************
# Tag name for the run (one word)                                    *
#*********************************************************************
  tag_1     = run_tag ! name of the run 
#*********************************************************************
# Number of events and rnd seed                                      *
# Warning: Do not generate more than 1M events in a single run       *
#*********************************************************************
  100000    = nevents ! Number of unweighted events requested
  0   = iseed   ! rnd seed (0=assigned automatically=default))
#*********************************************************************
# Collider type and energy                                           *
# lpp: 0=No PDF, 1=proton, -1=antiproton,                            *
#                2=elastic photon of proton/ion beam                 *
#             +/-3=PDF of electron/positron beam                     *
#             +/-4=PDF of muon/antimuon beam                         *
#*********************************************************************
     1        = lpp1    ! beam 1 type 
     1        = lpp2    ! beam 2 type
  6500.0    = ebeam1 ! beam 1 total energy in GeV
  6500.0    = ebeam2 ! beam 2 total energy in GeV
# To see polarised beam options: type "update beam_pol"

#*********************************************************************
# PDF CHOICE: this automatically fixes alpha_s and its evol.         *
# pdlabel: lhapdf=LHAPDF (installation needed) [1412.7420]           *
#          iww=Improved Weizsaecker-Williams Approx.[hep-ph/9310350] *
#          eva=Effective W/Z/A Approx.       [2111.02442]            *
#          edff=EDFF in gamma-UPC            [eq.(11) in 2207.03012] *
#          chff=ChFF in gamma-UPC            [eq.(13) in 2207.03012] *
#          none=No PDF, same as lhapdf with lppx=0                   *
#*********************************************************************
     lhapdf    = pdlabel     ! PDF set 
     303400    = lhaid     ! if pdlabel=lhapdf, this is the lhapdf number
# To see heavy ion options: type "update ion_pdf"
#*********************************************************************
# Renormalization and factorization scales                           *
#*********************************************************************
 True  = fixed_ren_scale  ! if .true. use fixed ren scale
 True = fixed_fac_scale  ! if .true. use fixed fac scale
 91.15348  = scale            ! fixed ren scale
 91.15348  = dsqrt_q2fact1    ! fixed fact scale for pdf1
 91.15348  = dsqrt_q2fact2    ! fixed fact scale for pdf2
 -1 = dynamical_scale_choice ! Choose one of the preselected dynamical choices
 1.0  = scalefact        ! scale factor for event-by-event scales

 
#*********************************************************************
# Type and output format
#*********************************************************************
  False     = gridpack  !True = setting up the grid pack
  -1.0 = time_of_flight ! threshold (in mm) below which the invariant livetime is not written (-1 means not written)
  average =  event_norm       ! average/sum. Normalization of the weight in the LHEF
# To see MLM/CKKW  merging options: type "update MLM" or "update CKKW"

#*********************************************************************
#
#*********************************************************************
# Phase-Space Optimization strategy (basic options)
#*********************************************************************
   0  = nhel          ! using helicities importance sampling or not.
                             ! 0: sum over helicity, 1: importance sampling
   1  = sde_strategy  ! default integration strategy (hep-ph/2021.00773)
                             ! 1 is old strategy (using amp square)
			     ! 2 is new strategy (using only the denominator)
# To see advanced option for Phase-Space optimization: type "update psoptim"			     
#*********************************************************************
# Customization (custom cuts/scale/bias/...)                         *
# list of files containing fortran function that overwrite default   *
#*********************************************************************
  = custom_fcts ! List of files containing user hook function
#*******************************                                                 
# Parton level cuts definition *
#*******************************
  0.0  = dsqrt_shat ! minimal shat for full process
#                                                                    
#
#*********************************************************************
# BW cutoff (M+/-bwcutoff*Gamma) ! Define on/off-shell for "$" and decay  
#*********************************************************************
  15.0  = bwcutoff      ! (M+/-bwcutoff*Gamma)
#*********************************************************************
# Standard Cuts                                                      *
#*********************************************************************
# Minimum and maximum pt's (for max, -1 means no cut)                *
#*********************************************************************
 0.0  = ptl       ! minimum pt for the charged leptons 
 -1.0  = ptlmax    ! maximum pt for the charged leptons
 {} = pt_min_pdg ! pt cut for other particles (use pdg code). Applied on particle and anti-particle
 {}	= pt_max_pdg ! pt cut for other particles (syntax e.g. {6: 100, 25: 50}) 
#
# For display option for energy cut in the partonic center of mass frame type 'update ecut'
#
#*********************************************************************
# Maximum and minimum absolute rapidity (for max, -1 means no cut)   *
#*********************************************************************
 -1.0  = etal    ! max rap for the charged leptons 
 0.0  = etalmin ! min rap for the charged leptons
 {} = eta_min_pdg ! rap cut for other particles (use pdg code). Applied on particle and anti-particle
 {} = eta_max_pdg ! rap cut for other particles (syntax e.g. {6: 2.5, 23: 5})
#*********************************************************************
# Minimum and maximum DeltaR distance                                *
#*********************************************************************
 0.0 = drll    ! min distance between leptons 
 -1.0  = drllmax ! max distance between leptons
#*********************************************************************
# Minimum and maximum invariant mass for pairs                       *
 # WARNING: for four lepton final state mmll cut require to have      *
 #          different lepton masses for each flavor!                  *           
#*********************************************************************
 81.0   = mmll    ! min invariant mass of l+l- (same flavour) lepton pair
 101.0  = mmllmax ! max invariant mass of l+l- (same flavour) lepton pair
 {} = mxx_min_pdg ! min invariant mass of a pair of particles X/X~ (e.g. {6:250})
 {'default': False} = mxx_only_part_antipart ! if True the invariant mass is applied only 
                       ! to pairs of particle/antiparticle and not to pairs of the same pdg codes.  
 #*********************************************************************
 # Minimum and maximum invariant mass for all letpons                 *
 #*********************************************************************
 0.0   = mmnl    ! min invariant mass for all letpons (l+- and vl) 
 -1.0  = mmnlmax ! max invariant mass for all letpons (l+- and vl) 
 #*********************************************************************
 # Minimum and maximum pt for 4-momenta sum of leptons / neutrino     *
 #  for pair of lepton includes only same flavor, opposite charge
 #*********************************************************************
 0.0   = ptllmin  ! Minimum pt for 4-momenta sum of leptons(l and vl)
 -1.0  = ptllmax  ! Maximum pt for 4-momenta sum of leptons(l and vl)
#*********************************************************************
# Inclusive cuts                                                     *
#*********************************************************************
 0.0  = xptl ! minimum pt for at least one charged lepton 
 #*********************************************************************
 # Control the pt's of leptons sorted by pt                           *
 #*********************************************************************
 0.0   = ptl1min ! minimum pt for the leading lepton in pt
 0.0   = ptl2min ! minimum pt for the second lepton in pt
 0.0   = ptl3min ! minimum pt for the third lepton in pt
 0.0   = ptl4min ! minimum pt for the fourth lepton in pt
 -1.0  = ptl1max ! maximum pt for the leading lepton in pt 
 -1.0  = ptl2max ! maximum pt for the second lepton in pt
 -1.0  = ptl3max ! maximum pt for the third lepton in pt
 -1.0  = ptl4max ! maximum pt for the fourth lepton in pt
#*********************************************************************
# maximal pdg code for quark to be considered as a light jet         *
# (otherwise b cuts are applied)                                     *
#*********************************************************************
 5 = maxjetflavor    ! Maximum jet pdg code
#*********************************************************************
#
#*********************************************************************
# Store info for systematics studies                                 *
# WARNING: Do not use for interference type of computation           *
#*********************************************************************
   False  = use_syst      ! Enable systematics studies
#
systematics = none ! none, systematics [python], SysCalc [depreceted, C++]
['--mur=0.5,1,2', '--muf=0.5,1,2', '--pdf=errorset'] = systematics_arguments ! see: https://cp3.irmp.ucl.ac.be/projects/madgraph/wiki/Systematics#Systematicspythonmodule

"""

# Edit param_card.dat
param_card_path = os.path.join(process_dir, "Cards", "param_card.dat")
new_param_card = """
######################################################################
## PARAM_CARD AUTOMATICALY GENERATED BY MG5                       ####
######################################################################
###################################
## INFORMATION FOR MASS
###################################
BLOCK MASS # 
      5 0.000000e+00 # mb
      6 1.720000e+02 # mt
      15 0.000000e+00 # mta
      23 9.115348e+01 # mz
      24 8.035197e+01 # mw
      25 1.250000e+02 # mh
      1 0.000000e+00 # d : 0.0
      2 0.000000e+00 # u : 0.0
      3 0.000000e+00 # s : 0.0
      4 0.000000e+00 # c : 0.0
      11 0.000000e+00 # e- : 0.0
      12 0.000000e+00 # ve : 0.0
      13 0.000000e+00 # mu- : 0.0
      14 0.000000e+00 # vm : 0.0
      16 0.000000e+00 # vt : 0.0
      21 0.000000e+00 # g : 0.0
      22 0.000000e+00 # a : 0.0
###################################
## INFORMATION FOR SMINPUTS
###################################
BLOCK SMINPUTS # 
      1 1.325070e+02 # aewm1
      3 1.180022e-01 # as (note that parameter not used if you use a pdf set)
###################################
## INFORMATION FOR DECAY
###################################
DECAY 6 1.470800e+00 # wt
DECAY 23 2.494266e+00 # wz
DECAY 24 2.084300e+00 # ww
DECAY 25 4.088000e-03 # wh
DECAY 1 0.000000e+00 # d : 0.0
DECAY 2 0.000000e+00 # u : 0.0
DECAY 3 0.000000e+00 # s : 0.0
DECAY 4 0.000000e+00 # c : 0.0
DECAY 5 0.000000e+00 # b : 0.0
DECAY 11 0.000000e+00 # e- : 0.0
DECAY 12 0.000000e+00 # ve : 0.0
DECAY 13 0.000000e+00 # mu- : 0.0
DECAY 14 0.000000e+00 # vm : 0.0
DECAY 15 0.000000e+00 # ta- : 0.0
DECAY 16 0.000000e+00 # vt : 0.0
DECAY 21 0.000000e+00 # g : 0.0
DECAY 22 0.000000e+00 # a : 0.0
"""
with open(param_card_path, 'w') as file:
    file.write(new_param_card)

# Path to Fortran source file
fortran_dummy_fct = os.path.join(process_dir, "SubProcesses", "dummy_fct.f")

new_dummy_fct = """
      logical FUNCTION dummy_cuts(P)
C**************************************************************************
C     INPUT:
C            P(0:3,1)           MOMENTUM OF INCOMING PARTON
C            P(0:3,2)           MOMENTUM OF INCOMING PARTON
C            P(0:3,3)           MOMENTUM OF ...
C            ALL MOMENTA ARE IN THE REST FRAME!!
C            COMMON/JETCUTS/   CUTS ON JETS
C     OUTPUT:
C            TRUE IF EVENTS PASSES ALL CUTS LISTED
C**************************************************************************
      IMPLICIT NONE
c
c     Constants
c
      include 'genps.inc'
      include 'nexternal.inc'
C
C     ARGUMENTS
C
      REAL*8 P(0:3,nexternal)
C
C     PARAMETERS
C
      real*8 PI
      parameter( PI = 3.14159265358979323846d0 )
c
c     particle identification
c
      LOGICAL  IS_A_J(NEXTERNAL),IS_A_L(NEXTERNAL)
      LOGICAL  IS_A_B(NEXTERNAL),IS_A_A(NEXTERNAL),IS_A_ONIUM(NEXTERNAL)
      LOGICAL  IS_A_NU(NEXTERNAL),IS_HEAVY(NEXTERNAL)
      logical  do_cuts(nexternal)
      COMMON /TO_SPECISA/IS_A_J,IS_A_A,IS_A_L,IS_A_B,IS_A_NU,IS_HEAVY,
     & IS_A_ONIUM, do_cuts
        
      real*8 E_total, px_total, py_total, pz_total, M_squared, p_z1(0:3),
     &   p_diboson(0:3), z1_boosted(0:3), cos_psi, dotp3p
      real*8 z_axis(3) 
      integer i

      E_total = 0.0d0
      px_total = 0.0d0
      py_total = 0.0d0
      pz_total = 0.0d0
      dummy_cuts = .false.
      z_axis(1) = 0.0d0
      z_axis(2) = 0.0d0
      z_axis(3) = 1.0d0

          
      do i = 3, 6      
         E_total = E_total + P(0,i)
         px_total = px_total + P(1,i)
         py_total = py_total + P(2,i)
         pz_total = pz_total + P(3,i)
         p_diboson = p_diboson + P(0:3, i)
      end do
      M_squared = E_total**2 - (px_total**2 + py_total**2 + pz_total**2)
      M_squared = dsqrt(M_squared)
        
      p_z1 = P(0:3, 3) + P(0:3, 4)
      call boostinvp(p_z1, p_diboson, z1_boosted)

      cos_psi = dotp3p(z1_boosted(1:3), z_axis)
     &           /(sqrt(dotp3p(z1_boosted(1:3), z1_boosted(1:3)))) 
      if (M_squared.gt.(900.0d0) .and. M_squared.lt.(950.0d0) .and.
     &   cos_psi.lt.(0.8d0) .and. cos_psi.gt.(0.7d0)) then   
              dummy_cuts = .true.
              return
      end if
      return
      end
        

      function dotp3p(p1,p2)
!     three-vector scalar product
!     adapted from Powheg-Box-Res
      implicit none
      real*8 dotp3p,p1(3),p2(3)
      dotp3p = p1(1)*p2(1)+p1(2)*p2(2)+p1(3)*p2(3)
      return
      end



      subroutine boostinvp(q,pboost,qprime)
!     boost routine adapted from Phantom MC
!     Comput. Phys. Commun. 180 (2009) 401-417
!     [arXiv:0801.335]
      implicit none
      real*8 rmboost,aux,aaux
      real*8 q(0:3), pboost(0:3), qprime(0:3)
      rmboost=pboost(0)**2-pboost(1)**2-pboost(2)**2-pboost(3)**2
      rmboost=sqrt(max(rmboost,0.0d0))
      aux=(q(0)*pboost(0)-q(1)*pboost(1)-q(2)*pboost(2)-q(3)*pboost(3))
     &     /rmboost
      aaux=(aux+q(0))/(pboost(0)+rmboost)
      qprime(0)=aux
      qprime(1)=q(1)-aaux*pboost(1)
      qprime(2)=q(2)-aaux*pboost(2)
      qprime(3)=q(3)-aaux*pboost(3)
      return
      end    

      subroutine get_dummy_x1(sjac, X1, R, pbeam1, pbeam2, stot, shat)
      implicit none
      include 'maxparticles.inc'
      include 'run.inc'
c      include 'genps.inc'
      double precision sjac ! jacobian. should be updated not reinit
      double precision X1   ! bjorken X. output
      double precision R    ! random value after grid transfrormation. between 0 and 1
      double precision pbeam1(0:3) ! momentum of the first beam (input and/or output)
      double precision pbeam2(0:3) ! momentum of the second beam (input and/or output)
      double precision stot        ! total energy  (input and /or output)
      double precision shat        ! output

c     global variable to set (or not)
      double precision cm_rap
      logical set_cm_rap
      common/to_cm_rap/set_cm_rap,cm_rap
      
      set_cm_rap=.false. ! then cm_rap will be set as .5d0*dlog(xbk(1)*ebeam(1)/(xbk(2)*ebeam(2)))
                         ! ebeam(1) and ebeam(2) are defined here thanks to 'run.inc'
      shat = x1*ebeam(1)*ebeam(2)
      return 
      end

      subroutine get_dummy_x1_x2(sjac, X, R, pbeam1, pbeam2, stot,shat)
      implicit none
      include 'maxparticles.inc'
      include 'run.inc'
c      include 'genps.inc'
      double precision sjac ! jacobian. should be updated not reinit
      double precision X(2)   ! bjorken X. output
      double precision R(2)    ! random value after grid transfrormation. between 0 and 1
      double precision pbeam1(0:3) ! momentum of the first beam
      double precision pbeam2(0:3) ! momentum of the second beam
      double precision stot        ! total energy
      double precision shat        ! output

c     global variable to set (or not)
      double precision cm_rap
      logical set_cm_rap
      common/to_cm_rap/set_cm_rap,cm_rap
      
      set_cm_rap=.false. ! then cm_rap will be set as .5d0*dlog(xbk(1)*ebeam(1)/(xbk(2)*ebeam(2)))
                         ! ebeam(1) and ebeam(2) are defined here thanks to 'run.inc'
      shat = x(1)*x(2)*ebeam(1)*ebeam(2)
      return 
      end


      logical  function dummy_boostframe()
      implicit none
c
c      
      dummy_boostframe = .false.
      return
      end
      

      double precision function user_dynamical_scale(P)
c     allow to define your own dynamical scale, need to set dynamical_scale_choice to 0 (or 10) to use it
      implicit none
      include 'nexternal.inc'
      double precision P(0:3, nexternal)
c     Commmon to have access to all variable defined in the run_card      
      include 'genps.inc'
      include 'run.inc'
      write(0,*) "dynamical scale set to 0"
      write(0,*) "need to be defined via user_hook method"
      stop 1
c     fixed scale
      return
      end

      
      
C ************************************************************
C default for the library implementing a dummy bias function
C ************************************************************
      subroutine bias_wgt_custom(p, original_weight, bias_weight)
      implicit none
C
C Parameters
C
          include 'nexternal.inc'

C     
C Arguments
C
          double precision p(0:3, nexternal)
          double precision original_weight, bias_weight
C
C local variables
C
C
C Global variables
C
C      common block with metadata for the bias
C
          double precision stored_bias_weight
c          data stored_bias_weight/1.0d0/
          logical impact_xsec, requires_full_event_info
C         Impact_xsec 
C         Not impacting the xsec since the bias is 1.0. Therefore
C         bias_wgt will not be written in the lhe event file.
C         Setting it to .True. makes sure that it will not be written.
C         Default: True
C         Requires_full_event_info          
C         Of course this module does not require the full event
C         information (color, resonances, helicities, etc..)
c         Default: False          
          common/bias/stored_bias_weight,impact_xsec,
     &                requires_full_event_info

C --------------------
C BEGIN IMPLEMENTATION
C --------------------
          bias_weight = 1.0d0

      return
      end subroutine bias_wgt_custom

"""

# Write the new Fortran file
with open(fortran_dummy_fct, 'w') as file:
    file.write(new_dummy_fct)


# Define regions of phase space for (0.0 to 1.0) and (200.0 to 1000.0)
regions = [
    [(cos_min, cos_min + 0.1), (mass_min, mass_min + 50.0)]
    for cos_min in [0.0 + 0.1 * i for i in range(9)]
    for mass_min in [300.0 + 50.0 * j for j in range(14)]
]


# Function to modify Fortran file
def modify_fortran_file(file_path, limits):
    # Read the file content
    with open(file_path, 'r') as file:
        content = file.read()

    # Prepare the regex pattern.
    # This pattern assumes the original statement is formatted as in the Vim substitution.
    # Adjust the whitespace or pattern if needed.
    pattern = (
        r"^\s+if \(M_squared\.gt\.\([^)]*\) \.and\. M_squared\.lt\.\([^)]*\) .and.\n"
        r"\s+&\s*cos_psi\.lt\.\([^)]*\) \.and\. cos_psi\.gt\.\([^)]*\)\) then"
    )

    # Build the replacement string using the provided limits
    replacement = (
        f"      if (M_squared.gt.({limits[1][0]}d0) .and. M_squared.lt.({limits[1][1]}d0) .and.\n"
        f"     &   cos_psi.lt.({limits[0][1]}d0) .and. cos_psi.gt.({limits[0][0]}d0)) then"
    )

    # Perform the substitution; re.DOTALL makes '.' match newline characters.
    new_content, count = re.subn(pattern, replacement, content, flags=re.DOTALL | re.MULTILINE)

    if count == 0:
        print("Pattern not found. No changes made.")
    else:
        # Write the file back
        with open(file_path, 'w') as file:
            file.write(new_content)
        print(f"File {file_path} successfully updated with limits {limits}")


# Modify the Fortran file for each region and run the process
for region in regions:
    modify_fortran_file(fortran_dummy_fct, region)
    # Run the MadGraph process using subprocess, streamlined
    subprocess.run(
        [os.path.join(process_dir, 'bin', 'generate_events')],
        input='0\n0\n', text=True,
        cwd=process_dir, check=True
    )
    print(f'Process for region {region} finished')