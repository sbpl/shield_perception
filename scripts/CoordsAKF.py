import numpy as np
from collections import deque
from filterpy.kalman import KalmanFilter

class CoordsAKF:
    def __init__(self,t0, initial_position):
        self.kf = KalmanFilter(dim_x=6, dim_z=3, dim_u=1)
        self.g = 9.81

        self.Q_list, self.R_list, self.P_list, self.t_QR = [], [], [], []
        self._riae_inited = False
        self.init_poition = initial_position
        self.t0 = float(t0)

    def init_velocity(self, t1, second_position):
        t1=float(t1)
        self.dt0 = t1 - self.t0
        self.init_velocity = (self.init_poition - second_position) / self.dt0
        
    def init_AKF(self):
        # Model: [x y z vx vy vz]
        self.kf.F = np.array([[1,0,0, self.dt0,0,0],
                              [0,1,0, 0,self.dt0,0],
                              [0,0,1, 0,0,self.dt0],
                              [0,0,0, 1,0,0],
                              [0,0,0, 0,1,0],
                              [0,0,0, 0,0,1]])
        self.kf.B = np.array([[0],[0],[0.5*self.dt0**2],[0],[0],[self.dt0]])
        self.kf.H = np.array([[1,0,0,0,0,0],
                              [0,1,0,0,0,0],
                              [0,0,1,0,0,0]])
        self.kf.Q = np.diag([0.01,0.01,0.01, 1.0,1.0,1.0])
        self.kf.R = np.eye(3)
        self.kf.P *= 100.0
        self.kf.x = np.hstack((self.init_poition, self.init_velocity)).reshape(6,1)
        self.riae_init_rest()

    def riae_init_rest(self, window_size=5, warmup_steps=6, alpha=0.05,
                  conf_level=0.99, adapt_Q=False, robust_update=True,
                  eig_clip=(1e-6, 1.0)):
        self._win = int(window_size)
        self._warm = int(warmup_steps)
        self._alpha = float(alpha)
        self._conf = float(conf_level)
        self._adapt_Q = bool(adapt_Q)
        self._robust_update = bool(robust_update)
        self._lam_min, self._lam_max = eig_clip
        self._chi2_1 = self._chi2_threshold(1, self._conf)

        self._innov_window = deque(maxlen=self._win)
        self._last_time = None
        self._i = 0  # step counter

        # clear logs
        self.Q_list.clear(); self.R_list.clear(); self.P_list.clear(); self.t_QR.clear()
        self._riae_inited = True

    def riae_step(self, t_k, z_k):
        """
        Process ONE measurement at time t_k with z_k=(x,y,z).
        Return current mean and state covariance matrix.
        """
        assert self._riae_inited, "Initiate RIAE AKF."
        z = np.asarray(z_k, float).reshape(3,1)
        H = self.kf.H

        if self._last_time is None:
            self._last_time = float(t_k)
            return self.kf.x.flatten()[:6], self.kf.P

        dt = float(t_k) - self._last_time
        self._last_time = float(t_k)

        self.kf.F = np.array([[1,0,0, dt,0,0],
                              [0,1,0, 0,dt,0],
                              [0,0,1, 0,0,dt],
                              [0,0,0, 1,0,0],
                              [0,0,0, 0,1,0],
                              [0,0,0, 0,0,1]])
        self.kf.B[-1] = [dt]
        self.kf.B[2]  = [0.5*dt**2]

        # ---------- predict with gravity ----------
        self.kf.predict(u=np.array([[-self.g]]))

        # ---------- innovation + per-axis soft χ² gating ----------
        innovation = z - (H @ self.kf.x)                   # 3x1
        S_pred     = H @ self.kf.P @ H.T + self.kf.R       # 3x3

        S_diag  = np.clip(np.diag(S_pred).reshape(-1,1), 1e-12, None)
        kappa_i = (innovation**2) / S_diag                 # 3x1, df=1 per axis
        innovation_rev = innovation.copy()
        mask = (kappa_i >= self._chi2_1)
        if np.any(mask):
            shrink = np.exp(-(kappa_i - self._chi2_1)/self._chi2_1)
            innovation_rev = np.where(mask, innovation * shrink, innovation)

        # buffer revised innovation for adaptation
        self._innov_window.append(innovation_rev)

        # ---------- adapt R after warmup ----------
        self._i += 1
        if (self._i >= self._warm) and (len(self._innov_window) == self._win):
            Sk = sum([e @ e.T for e in self._innov_window]) / self._win  # 3x3
            Pp = H @ self.kf.P @ H.T                                     # 3x3  (using P^- consistently)

            # R <- (1-α)R + α * clip(Sk - Pp)
            R_temp = self._sym_clip_psd(Sk - Pp)
            self.kf.R = self._sym_clip_psd((1 - self._alpha)*self.kf.R + self._alpha*R_temp)

            if self._adapt_Q:
                # Q <- (1-α)Q + α * K(Sk - Pp)K^T   with K = P H^T Sk^{-1}
                try:
                    Sk_inv = np.linalg.inv(Sk)
                except np.linalg.LinAlgError:
                    Sk_inv = np.linalg.inv(Sk + 1e-9*np.eye(Sk.shape[0]))
                K = self.kf.P @ H.T @ Sk_inv
                Q_temp = 0.5 * (K @ (Sk - Pp) @ K.T + (K @ (Sk - Pp) @ K.T).T)
                Q_temp = self._sym_clip_psd(Q_temp)
                self.kf.Q = self._sym_clip_psd((1 - self._alpha)*self.kf.Q + self._alpha*Q_temp)

            # Logs
            self.R_list.append(self.kf.R.copy())
            self.Q_list.append(self.kf.Q.copy())
            self.P_list.append(self.kf.P.copy())
            self.t_QR.append(self._last_time)

        # ---------- measurement update ----------
        if self._robust_update:
            z = (H @ self.kf.x) + innovation_rev
        self.kf.update(z)
        return self.kf.x.flatten()[:6], self.kf.P
    
    def kf_predict(self, dt, cal_traj, num_future_steps):
        """
        This is the prediction function to predict object's future states within certain time period.
        dt: delta t which is the time step 
        num_future_steps: how may future steps it has
        dt*num_future_steps: how long it is after the current time.
        """
        self.kf.F = np.array([[1, 0, 0, dt, 0, 0],
                        [0, 1, 0, 0, dt, 0],
                        [0, 0, 1, 0, 0, dt],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1]])
        self.kf.B = np.array([[0], [0], [0.5 * dt**2], [0], [0], [dt]])
        for i in range(num_future_steps):
            self.kf.predict(u=np.array([[-self.g]]))
            cal_traj = np.vstack((cal_traj, [self.kf.x[0, 0], self.kf.x[1, 0], self.kf.x[2, 0]]))

    @staticmethod
    def _chi2_threshold(df, conf=0.99):
        table = { (1,0.95):3.841459, (1,0.99):6.634897,
                  (3,0.95):7.814728, (3,0.99):11.344867 }
        return table.get((df, conf), float('inf'))

    def _sym_clip_psd(self, M):
        M = 0.5*(M + M.T)
        evals, evecs = np.linalg.eigh(M)
        evals = np.clip(evals, self._lam_min, self._lam_max)
        return evecs @ np.diag(evals) @ evecs.T
