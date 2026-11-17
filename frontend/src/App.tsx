import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';

// Auth
import LoginPage from './pages/auth/LoginPage';
import FirstTimeLoginPage from './pages/auth/FirstTimeLoginPage';
import SetPasswordPage from './pages/auth/SetPasswordPage';

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

        {/* Staff */}
        <Route path="/staff/home"           element={<StaffHomePage />} />
        <Route path="/staff/review-history" element={<ReviewHistoryPage />} />
        <Route path="/staff/search"         element={<SearchPage />} />
        <Route path="/staff/hotspots"       element={<HotspotsPage />} />
        <Route path="/staff/cameras"        element={<StaffCamerasPage />} />

        {/* Admin */}
        <Route path="/admin/home"     element={<AdminHomePage />} />
        <Route path="/admin/workload" element={<WorkloadSummaryPage />} />
        <Route path="/admin/rules"    element={<MonitoringRulesPage />} />
        <Route path="/admin/cameras"  element={<AdminCamerasPage />} />

        {/* Default */}
        <Route path="*" element={<Navigate to="/login" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
