!===============================================================
! Program cpwt_spectra
!
!   Calculates local psd functions of 2-D fields 
!   with a wavelet tranform method. Directional information 
!   is averaged to yield isotropic (1D) psd.
!
!
!   input:
!          file "params.com" -> grid parameters 
!          file "boug.out.txt"  -> Bouguer gravity data set
!          file "retopo.out.txt"  -> Rock-equivalent topography data set
!          file "fair.out.txt"  -> Free air gravity data set
!
!   output:
!	       file "spectrum.[bg,tp,fa].* -> local psd of Bouguer, topo or free air
!
!   Run: cpwt_spectra
!
!===============================================================
!   Pascal Audet
!   Department of Earth and Planetary Science
!   UC Berkeley
!   paudet@berkeley.edu
!
!   July 2009
!================================================================
!
!   Modified August 7, 2019
!
!   Pascal Audet
!   Department of Earth Sciences
!   University of Ottawa
!   pascal.audet@uottawa.ca
!
!================================================================

   
      MODULE conf

      IMPLICIT NONE

      REAL, PARAMETER :: pi = 3.141592653589793
      INTEGER, PARAMETER :: na = 12

      REAL :: k0

      END MODULE conf



      MODULE cpwt


      CONTAINS




      SUBROUTINE wlet_transform(nx, ny, nnx, nny, dx, dy, ns, kf, &
        grid, wt_grid)     

      USE conf

      IMPLICIT NONE

      INTEGER :: nx, ny, nnx, nny, ns, nn(2)
      REAL :: dx, dy
      REAL :: grid(nx,ny), grid_pad(nnx,nny), kf(ns)
      COMPLEX :: wt_grid(ns,na,nx,ny)
      REAL :: kx(nnx), ky(nny)
      COMPLEX :: ft_grid(nnx,nny), ft2_grid(nnx,nny), daughter(nnx,nny) 

      INTEGER :: is, ia
      REAL :: da, scales, angle, lam, kk
!
! Python bindings
!
!f2py INTEGER, intent(in) :: nx, ny, nnx, nny
!f2py REAL, intent(in) :: dx, dy
!f2py INTEGER, intent(in) :: ns
!f2py REAL, intent(in) :: kf, grid
!f2py COMPLEX, intent(out) :: wt_grid

        grid_pad = 0.
!
! Remove mean value
!
        grid_pad(1:nx,1:ny) = grid - sum(grid)/nx/ny
!
! Complex form
!
        ft_grid = CMPLX(grid_pad)
!
! Fourier transform
!
        nn(1) = nnx
        nn(2) = nny
        CALL fourn(ft_grid,nn,2,1)
!
! Define wavenumbers
!
        CALL defk(nnx,nny,dx,dy,kx,ky)
!
! Define angle parameters
!
        da = REAL(pi/na)
!_______________________________________________
!     
!     MAIN WAVELET TANSFORM LOOP
!_______________________________________________
!
!
! Loop through scales
!
        DO is = 1,ns
!
! Define wavenumbers and scales
!
          kk = kf(is)*1.e3
          lam = 2.*pi/kk
          scales = k0/kk
          ! PRINT*,'scale = ',scales,'wavelength [km] = ',lam
!
! Loop through angles
!
          DO ia = 1,na
!
! Define angle increments
! 
            angle = REAL(ia-1)*da-pi/2.e0
            ! PRINT*,ia,'angle = ',angle*180/pi
!
! Calculate daughter wavelet
!
            CALL wave_function(nnx,nny,kx,ky,scales,angle,daughter)
!
! Compute wavelet transform in Fourier space
!
            daughter = daughter*CONJG(ft_grid)
!
! Back to physical space
!
            CALL fourn(daughter,nn,2,-1)
!
! Normalization
!
            ft2_grid = (daughter)/FLOAT(nnx*nny)
!
! Quadrant swap
!
            CALL cfftswap(ft2_grid,nnx,nny)
!
! Store arrays
!
            wt_grid(:,:,ia,is) = ft2_grid(1:nx,1:ny)

          END DO
        END DO

      END SUBROUTINE wlet_transform

      END MODULE cpwt