! ***************************************************************************************
! subroutine bl_avg_fc_to_cc 
! ***************************************************************************************

subroutine bl_avg_fc_to_cc (lo, hi, &
     cc, ccl1, ccl2, ccl3, cch1, cch2, cch3, &
     fx, fxl1, fxl2, fxl3, fxh1, fxh2, fxh3, &
     fy, fyl1, fyl2, fyl3, fyh1, fyh2, fyh3, &
     fz, fzl1, fzl2, fzl3, fzh1, fzh2, fzh3, &
     dx, problo, coord_type)

  implicit none
  integer          :: lo(3),hi(3), coord_type
  integer          :: ccl1, ccl2, ccl3, cch1, cch2, cch3
  integer          :: fxl1, fxl2, fxl3, fxh1, fxh2, fxh3
  integer          :: fyl1, fyl2, fyl3, fyh1, fyh2, fyh3
  integer          :: fzl1, fzl2, fzl3, fzh1, fzh2, fzh3
  double precision :: cc(ccl1:cch1, ccl2:cch2, ccl3:cch3, 3)
  double precision :: fx(fxl1:fxh1, fxl2:fxh2, fxl3:fxh3)
  double precision :: fy(fyl1:fyh1, fyl2:fyh2, fyl3:fyh3)
  double precision :: fz(fzl1:fzh1, fzl2:fzh2, fzl3:fzh3)
  double precision :: dx(3), problo(3)

  ! Local variables
  integer          :: i,j,k
  
  do k=lo(3),hi(3)
     do j=lo(2),hi(2)
        do i=lo(1),hi(1)
           cc(i,j,k,1) = 0.5d0 * ( fx(i,j,k) + fx(i+1,j,k) )
           cc(i,j,k,2) = 0.5d0 * ( fy(i,j,k) + fy(i,j+1,k) )
           cc(i,j,k,3) = 0.5d0 * ( fz(i,j,k) + fz(i,j,k+1) )
        enddo
     enddo
  enddo

end subroutine bl_avg_fc_to_cc

! ***************************************************************************************
! subroutine bl_avg_cc_to_fc 
! ***************************************************************************************

subroutine bl_avg_cc_to_fc (xlo, xhi, ylo, yhi, zlo, zhi, &
     fx, fxl1, fxl2, fxl3, fxh1, fxh2, fxh3, &
     fy, fyl1, fyl2, fyl3, fyh1, fyh2, fyh3, &
     fz, fzl1, fzl2, fzl3, fzh1, fzh2, fzh3, &
     cc, ccl1, ccl2, ccl3, cch1, cch2, cch3, &
     dx, problo, coord_type)

  implicit none
  integer          :: xlo(3),xhi(3), ylo(3),yhi(3), zlo(3), zhi(3), coord_type
  integer          :: fxl1, fxl2, fxl3, fxh1, fxh2, fxh3
  integer          :: fyl1, fyl2, fyl3, fyh1, fyh2, fyh3
  integer          :: fzl1, fzl2, fzl3, fzh1, fzh2, fzh3
  integer          :: ccl1, ccl2, ccl3, cch1, cch2, cch3
  double precision :: cc(ccl1:cch1, ccl2:cch2, ccl3:cch3)
  double precision :: fx(fxl1:fxh1, fxl2:fxh2, fxl3:fxh3)
  double precision :: fy(fyl1:fyh1, fyl2:fyh2, fyl3:fyh3)
  double precision :: fz(fzl1:fzh1, fzl2:fzh2, fzl3:fzh3)
  double precision :: dx(3), problo(3)

  ! Local variables
  integer          :: i,j,k
  
  do k=xlo(3),xhi(3)
     do j=xlo(2),xhi(2)
        do i=xlo(1),xhi(1)
           fx(i,j,k) = 0.5d0 * (cc(i-1,j,k) + cc(i,j,k))
        enddo
     enddo
  end do

  do k=ylo(3),yhi(3)
     do j=ylo(2),yhi(2)
        do i=ylo(1),yhi(1)
           fy(i,j,k) = 0.5d0 * (cc(i,j-1,k) + cc(i,j,k))
        enddo
     enddo
  end do

  do k=zlo(3),zhi(3)
     do j=zlo(2),zhi(2)
        do i=zlo(1),zhi(1)
           fz(i,j,k) = 0.5d0 * (cc(i,j,k-1) + cc(i,j,k))
        enddo
     enddo
  end do

end subroutine bl_avg_cc_to_fc

! ***************************************************************************************
! subroutine bl_avgdown_faces
! ***************************************************************************************

subroutine bl_avgdown_faces (lo, hi, &
     f, f_l1, f_l2, f_l3, f_h1, f_h2, f_h3, &
     c, c_l1, c_l2, c_l3, c_h1, c_h2, c_h3, &
     ratio,idir)

  implicit none
  integer          :: lo(3),hi(3)
  integer          :: f_l1, f_l2, f_l3, f_h1, f_h2, f_h3
  integer          :: c_l1, c_l2, c_l3, c_h1, c_h2, c_h3
  integer          :: ratio(3), idir
  double precision :: f(f_l1:f_h1, f_l2:f_h2, f_l3:f_h3)
  double precision :: c(c_l1:c_h1, c_l2:c_h2, c_l3:c_h3)

  ! Local variables
  integer i,j,k,m,n,facx,facy,facz

  facx = ratio(1)
  facy = ratio(2)
  facz = ratio(3)

  if (idir .eq. 0) then
         do k = lo(3), hi(3)
            do j = lo(2), hi(2)
               do i = lo(1), hi(1)
                  c(i,j,k) = 0.d0
                  do n = 0,facy-1
                     do m = 0,facz-1
                        c(i,j,k) = c(i,j,k) + f(facx*i,facy*j+n,facz*k+m)
                     end do
                  end do
                  c(i,j,k) = c(i,j,k) / (facy*facz)
               end do
            end do
         end do
  else if (idir .eq. 1) then
         do k = lo(3), hi(3)
            do j = lo(2), hi(2)
               do i = lo(1), hi(1)
                  c(i,j,k) = 0.d0
                  do n = 0,facx-1
                     do m = 0,facz-1
                        c(i,j,k) = c(i,j,k) + f(facx*i+n,facy*j,facz*k+m)
                     end do
                  end do
                  c(i,j,k) = c(i,j,k) / (facx*facz)
               end do
            end do
         end do
  else
         do k = lo(3), hi(3)
            do j = lo(2), hi(2)
               do i = lo(1), hi(1)
                  c(i,j,k) = 0.d0
                  do n = 0,facx-1
                     do m = 0,facy-1
                        c(i,j,k) = c(i,j,k) + f(facx*i+n,facy*j+m,facz*k)
                     end do
                  end do
                  c(i,j,k) = c(i,j,k) / (facx*facy)
               end do
            end do
         end do
  end if

end subroutine bl_avgdown_faces

! ***************************************************************************************
! subroutine bl_avgdown
! ***************************************************************************************

      subroutine bl_avgdown (lo,hi,&
                             fine,f_l1,f_l2,f_l3,f_h1,f_h2,f_h3, &
                             crse,c_l1,c_l2,c_l3,c_h1,c_h2,c_h3, &
                             lrat,ncomp)

      implicit none

      integer f_l1,f_l2,f_l3,f_h1,f_h2,f_h3
      integer c_l1,c_l2,c_l3,c_h1,c_h2,c_h3
      integer lo(3), hi(3)
      integer lrat(3), ncomp
      double precision crse(c_l1:c_h1,c_l2:c_h2,c_l3:c_h3,ncomp)
      double precision fine(f_l1:f_h1,f_l2:f_h2,f_l3:f_h3,ncomp)

      integer i, j, k, ic, jc, kc, ioff, joff, koff
      integer clo(3),chi(3)
      double precision volfrac

      clo(1:3) = lo(1:3) / lrat(1:3) 
      chi(1:3) = hi(1:3) / lrat(1:3) 

      !
      ! ::::: set coarse grid to zero on overlap
      !
      do kc = clo(3), chi(3)
         do jc = clo(2), chi(2)
            do ic = clo(1), chi(1)
               crse(ic,jc,kc,:) = 0.d0
            enddo
         enddo
      enddo
      !
      ! ::::: sum fine data
      !
      do koff = 0, lrat(3)-1
        do kc = clo(3), chi(3)
          k = kc*lrat(3) + koff
          do joff = 0, lrat(2)-1
            do jc = clo(2), chi(2)
              j = jc*lrat(2) + joff
              do ioff = 0, lrat(1)-1
                do ic = clo(1), chi(1)
                  i = ic*lrat(1) + ioff
                  crse(ic,jc,kc,1:ncomp) = crse(ic,jc,kc,1:ncomp) + fine(i,j,k,1:ncomp)
                enddo
              enddo
            enddo
          enddo
        enddo
      enddo

      volfrac = 1.d0/dble(lrat(1)*lrat(2)*lrat(3))
      do kc = clo(3), chi(3)
         do jc = clo(2), chi(2)
            do ic = clo(1), chi(1)
               crse(ic,jc,kc,1:ncomp) = volfrac*crse(ic,jc,kc,1:ncomp)
            enddo
         enddo
      enddo

      end subroutine bl_avgdown
! ***************************************************************************************
