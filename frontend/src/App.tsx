import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { RequireAuth } from './components/layout/RequireAuth';

// Auth
import LoginPage from './pages/auth/LoginPage';
import FirstTimeLoginPage from './pages/auth/FirstTimeLoginPage';
import SetPasswordPage from './pages/auth/SetPasswordPage';
import ForgotPasswordPage from './pages/auth/ForgotPasswordPage';
import ResetPasswordPage from './pages/auth/ResetPasswordPage';

// Staff
import StaffHomePage from './pages/staff/StaffHomePage';
import ReviewHistoryPage from './pages/staff/ReviewHistoryPage';
import SearchPage from './pages/staff/SearchPage';
import HotspotsPage from './pages/staff/HotspotsPage';
import StaffCamerasPage from './pages/staff/CamerasPage';

// Admin
import AdminHomePage from './pages/admin/AdminHomePage';
import WorkloadSummaryPage from './pages/admin/WorkloadSummaryPage';
import MonitoringRulesPage from './pages/admin/MonitoringRulesPage';
import AdminCamerasPage from './pages/admin/AdminCamerasPage';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Auth */}
        <Route path="/login"        element={<LoginPage />} />
        <Route path="/first-login"  element={<FirstTimeLoginPage />} />
        <Route path="/set-password" element={<SetPasswordPage />} />
        <Route path="/forgot-password" element={<ForgotPasswordPage />} />
        <Route path="/reset-password"  element={<ResetPasswordPage />} />

        {/* Staff: anyone signed in */}
        <Route element={<RequireAuth />}>
          <Route path="/staff/home"           element={<StaffHomePage />} />
          <Route path="/staff/review-history" element={<ReviewHistoryPage />} />
          <Route path="/staff/search"         element={<SearchPage />} />
          <Route path="/staff/hotspots"       element={<HotspotsPage />} />
          <Route path="/staff/cameras"        element={<StaffCamerasPage />} />
        </Route>

        {/* Admin: admins only */}
        <Route element={<RequireAuth role="admin" />}>
          <Route path="/admin/home"     element={<AdminHomePage />} />
          <Route path="/admin/workload" element={<WorkloadSummaryPage />} />
          <Route path="/admin/rules"    element={<MonitoringRulesPage />} />
          <Route path="/admin/cameras"  element={<AdminCamerasPage />} />
        </Route>

        {/* Default */}
        <Route path="*" element={<Navigate to="/login" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
