      MODULE conf_flex

      IMPLICIT NONE

      DOUBLE PRECISION, PARAMETER :: pi = 3.141592653589793d0
      DOUBLE PRECISION, PARAMETER :: g = 9.81d0
      DOUBLE PRECISION, PARAMETER :: Em = 1.d11
      DOUBLE PRECISION, PARAMETER :: nu = 0.25d0
      DOUBLE PRECISION, PARAMETER :: Gc = 6.67d-6

      DOUBLE PRECISION :: rhoc, rhom, rhof, rhow, rhoa, wd, zc
      INTEGER :: water, boug

      END MODULE conf_flex



      MODULE flex

      CONTAINS

      SUBROUTINE flexfilter_top(ns, psi, filt)     

      USE conf_flex

      IMPLICIT NONE

      INTEGER :: ns
      DOUBLE PRECISION :: psi(ns), filt(ns)

        filt = -((rhoc-rhof)/(rhom-rhoc))*(1. + psi/(rhom-rhoc)/g)**(-1.)

        RETURN

      END SUBROUTINE flexfilter_top      


      SUBROUTINE flexfilter_bot(ns, psi, filt)     

      USE conf_flex

      IMPLICIT NONE

      INTEGER :: ns
      DOUBLE PRECISION :: psi(ns), filt(ns)

        filt = -((rhoc-rhof)/(rhom-rhoc))*(1. + psi/(rhoc-rhof)/g)

        RETURN

      END SUBROUTINE flexfilter_bot      


      SUBROUTINE decon(ns, theta, phi, k, A, mu_h, mu_w, nu_h, nu_w)

      USE conf_flex

      IMPLICIT NONE

      INTEGER :: ns
      DOUBLE PRECISION :: theta(ns), phi(ns), k(ns), A
      DOUBLE PRECISION :: mu_h(ns), mu_w(ns), nu_h(ns), nu_w(ns)

        mu_h = 1./(1.-theta)
        mu_w = 1./(phi-1.)
        nu_h = 2.*pi*Gc*(A*(rhoc-rhof)*EXP(-k*wd) + (rhom-rhoc)*theta*EXP(-k*(zc+wd)))
        nu_h = nu_h/(1.-theta)
        nu_w = 2.*pi*Gc*(A*(rhoc-rhof)*EXP(-k*wd) + (rhom-rhoc)*phi*EXP(-k*(zc+wd)))
        nu_w = nu_w/(phi-1.)

        RETURN

      END SUBROUTINE decon


      SUBROUTINE tr_func(ns, mu_h, mu_w, nu_h, nu_w, F, alpha, admit, coh)

      USE conf_flex

      IMPLICIT NONE

      INTEGER :: ns
      DOUBLE PRECISION :: mu_h(ns), mu_w(ns), nu_h(ns), nu_w(ns)
      DOUBLE PRECISION :: r, ff, F, alpha
      DOUBLE COMPLEX :: hg(ns)
      DOUBLE PRECISION :: hh(ns), gg(ns)
      DOUBLE COMPLEX :: admit(ns), cohy(ns)
      DOUBLE PRECISION :: coh(ns)
      DOUBLE COMPLEX :: j

        j = DCMPLX(0.,1.)
        r = (rhoc-rhof)/(rhom-rhoc)
        ff = F/(1. - F)
        hg = nu_h*mu_h + nu_w*mu_w*(ff**2)*(r**2.) &
              + (nu_h*mu_w + nu_w*mu_h)*ff*r*COS(alpha) &
              + j*(nu_h*mu_w - nu_w*mu_h)*ff*r*SIN(alpha)
        hh = mu_h**2 + (mu_w*ff*r)**2. + 2.*mu_h*mu_w*ff*r*COS(alpha)
        gg = nu_h**2 + (nu_w*ff*r)**2. + 2.*nu_h*nu_w*ff*r*COS(alpha)
        admit = hg/hh
        cohy = hg/SQRT(hh)/SQRT(gg)
        coh = REAL(cohy)**2.
  
        RETURN

      END SUBROUTINE tr_func


      SUBROUTINE real_xspec_functions(ns, k, Te, F, alpha, wdepth, admit, coh)

      USE conf_flex

      IMPLICIT NONE

      INTEGER :: ns
      DOUBLE PRECISION :: k(ns), Te, F, alpha, wdepth

      DOUBLE PRECISION :: A, D, psi(ns), theta(ns), phi(ns)
      DOUBLE PRECISION :: mu_h(ns), mu_w(ns), nu_h(ns), nu_w(ns)
      DOUBLE COMPLEX :: cadmit(ns)

      DOUBLE PRECISION :: admit(ns), coh(ns)
!
! Python bindings
!
!f2py DOUBLE PRECISION, intent(in) :: k
!f2py INTEGER, intent(hide),depend(k) :: ns=shape(k,0)
!f2py DOUBLE PRECISION, intent(in) :: Te, F, alpha, wd
!f2py DOUBLE PRECISION, intent(out) :: admit, coh

        IF (boug.eq.1) THEN
          A = 0.
        ELSE
          A = 1.
        END IF

        IF (water.eq.1) THEN
          rhof = rhow
          wd = wdepth*1.e3
        ELSE
          rhof = rhoa
          wd = 0.
        END IF

        ! Te in meters
        Te = Te*1.e3

        ! Flexural rigidity
        D = Em*Te**3/12./(1.-nu**2.)

        ! Isostatic function
        psi = D*k**4.

        ! Flexural filters
        CALL flexfilter_top(ns, psi, theta)
        CALL flexfilter_bot(ns, psi, phi)
        CALL decon(ns, theta, phi, k, A, mu_h, mu_w, nu_h, nu_w)

        ! Get spectral functions
        CALL tr_func(ns, mu_h, mu_w, nu_h, nu_w, F, alpha, cadmit, coh)

        ! Get real-valued admittance
        admit = REAL(cadmit)

        RETURN

      END SUBROUTINE real_xspec_functions

    END MODULE flex

