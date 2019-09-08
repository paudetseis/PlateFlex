!===============================================================
!
!================================================================

   
      MODULE conf

      IMPLICIT NONE

      REAL, PARAMETER :: pi = 3.141592653589793
      INTEGER, PARAMETER :: na = 23

      REAL :: k0

      END MODULE conf



      MODULE cpwt

      CONTAINS

      SUBROUTINE wlet_transform(grid, nx, ny, nnx, nny, dx, dy, kf, ns, &
        wt_grid)     

      USE conf

      IMPLICIT NONE

      INTEGER :: nx, ny, nnx, nny, ns, nn(2)
      REAL    :: dx, dy
      REAL    :: grid(nx,ny), grid_pad(nnx,nny), kf(ns)
      COMPLEX :: wt_grid(nx,ny,na,ns)
      REAL    :: kx(nnx), ky(nny)
      COMPLEX :: ft_grid(nnx,nny), ft2_grid(nnx,nny), daughter(nnx,nny) 

      INTEGER :: is, ia
      REAL    :: da, scales, angle, lam, kk
      CHARACTER(LEN=1) :: trail
      CHARACTER(LEN=30) :: progress
!
! Python bindings
!
!f2py REAL, intent(in) :: grid
!f2py INTEGER, intent(hide),depend(grid) :: nx=shape(grid,0), ny=shape(grid,1)
!f2py INTEGER, intent(in) :: nnx, nny
!f2py REAL, intent(in) :: dx, dy
!f2py REAL, intent(in) :: kf
!f2py INTEGER, intent(hide),depend(kf) :: ns=shape(kf,0)
!f2py COMPLEX, intent(out) :: wt_grid

        CALL taper_data(grid,nx,ny,10)
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
        trail = '-'
        progress = ''
        DO is = 1,ns
!
! Define wavenumbers and scales
!
          trail = '_'
          kk = kf(is)*1.e3
          lam = 2.*pi/kk
          scales = k0/kk
          PRINT*,is,' of ',ns,TRIM(progress)//trail
          progress = TRIM(progress)//trail
!
! Loop through angles
!
          DO ia = 1,na
!
! Define angle increments
! 
            angle = REAL(ia-1)*da-pi/2.e0
!
! Calculate daughter wavelet
!
            CALL wave_function(nnx,nny,kx,ky,k0,scales,angle,daughter)
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