// Demo data used when the incidents API is not reachable.
// The original mock module was not committed with the pages that import it;
// this one follows the shapes in ../types and the ids those pages expect.
import type { Camera, Incident, MonitoringRule, StaffMember } from '../types';
import { DEFAULT_RULES } from './rules';

function at(daysAgo: number, time: string): string {
  const day = new Date();
  day.setDate(day.getDate() - daysAgo);
  return `${day.toISOString().slice(0, 10)}T${time}`;
}

export const MOCK_STAFF: StaffMember[] = [
  { id: 's1', name: 'Jane Tan', email: 'jane.tan@example.com', initials: 'JT' },
  { id: 's2', name: 'Raj A.', email: 'raj.a@example.com', initials: 'RA' },
  { id: 's3', name: 'Sam Lee', email: 'sam.lee@example.com', initials: 'SL' },
];

export const MOCK_CAMERAS: Camera[] = [
  { id: 'cam-01', name: 'Cam 01', location: 'Meeting Room 1', status: 'online', rules: ['rule-01', 'rule-02', 'rule-08'] },
  { id: 'cam-02', name: 'Cam 02', location: 'Main Entrance', status: 'online', rules: ['rule-03'] },
  { id: 'cam-03', name: 'Cam 03', location: 'Corridor 2', status: 'online', rules: ['rule-04'] },
  { id: 'cam-04', name: 'Cam 04', location: 'Carpark B2', status: 'online', rules: ['rule-05'] },
  { id: 'cam-05', name: 'Cam 05', location: 'Lift Lobby 1', status: 'online', rules: ['rule-06'] },
  { id: 'cam-06', name: 'Cam 06', location: 'Staircase B', status: 'online', rules: ['rule-07', 'rule-09'] },
  { id: 'cam-07', name: 'Cam 07', location: 'Pantry L3', status: 'offline', rules: ['rule-10'], lastSeen: at(0, '18:12:00') },
  { id: 'cam-09', name: 'Cam 09', location: 'Loading Bay', status: 'offline', rules: [], lastSeen: at(1, '09:40:00') },
];

// The demo rules live in ./rules (DEFAULT_RULES), shared by the admin and staff screens.
export const MOCK_RULES: MonitoringRule[] = DEFAULT_RULES;

export const MOCK_INCIDENTS: Incident[] = [
  { id: 'inc-001', camId: 'cam-04', location: 'Carpark B2', type: 'Smoking', severity: 'Critical', dateTime: at(0, '23:18:06'), outcome: 'Pending', confidence: 0.92, description: 'Person smoking a cigarette next to the staircase door, ash dropped on the floor.', duration: '3 min 32 s' },
  { id: 'inc-002', camId: 'cam-01', location: 'Meeting Room 1', type: 'Littering', severity: 'Minor', dateTime: at(0, '10:15:00'), outcome: 'Pending', confidence: 0.81, description: 'Drink cans and wrappers left on the table after the room emptied.', duration: '12 min 05 s' },
  { id: 'inc-003', camId: 'cam-03', location: 'Corridor 2', type: 'Loitering', severity: 'Minor', dateTime: at(0, '11:02:00'), outcome: 'Pending', confidence: 0.74, description: 'One person standing by the fire exit without moving on.', duration: '6 min 40 s' },
  { id: 'inc-004', camId: 'cam-05', location: 'Lift Lobby 1', type: 'Littering', severity: 'Minor', dateTime: at(0, '20:37:00'), outcome: 'Pending', confidence: 0.68, description: 'A drink cup left on the floor beside the lift.', duration: '0 min 18 s' },
  { id: 'inc-005', camId: 'cam-06', location: 'Staircase B', type: 'Smoking', severity: 'Moderate', dateTime: at(1, '14:22:00'), outcome: 'Escalated', confidence: 0.88, description: 'Two people smoking on the landing.', assignedTo: 's1', remark: 'Repeat location this week.', duration: '4 min 10 s' },
  { id: 'inc-006', camId: 'cam-02', location: 'Main Entrance', type: 'Loitering', severity: 'Minor', dateTime: at(1, '02:05:00'), outcome: 'Dismissed', confidence: 0.61, description: 'Person waiting near the entrance.', assignedTo: 's2', remark: 'Delivery rider waiting for pickup.', duration: '7 min 02 s' },
  { id: 'inc-007', camId: 'cam-04', location: 'Carpark B2', type: 'Littering', severity: 'Moderate', dateTime: at(2, '17:48:00'), outcome: 'Accepted', confidence: 0.9, description: 'A bag of rubbish dumped between two parked cars.', assignedTo: 's1', duration: '21 min 30 s' },
  { id: 'inc-008', camId: 'cam-05', location: 'Lift Lobby 1', type: 'Littering', severity: 'Minor', dateTime: at(2, '09:12:00'), outcome: 'Accepted', confidence: 0.79, description: 'Food wrapper dropped while leaving the lift.', assignedTo: 's2', duration: '0 min 12 s' },
  { id: 'inc-009', camId: 'cam-01', location: 'Meeting Room 1', type: 'Smoking', severity: 'Moderate', dateTime: at(3, '19:30:00'), outcome: 'Dismissed', confidence: 0.66, description: 'Hand raised to the mouth repeatedly.', assignedTo: 's1', remark: 'Drinking from a bottle, not smoking.', duration: '1 min 05 s' },
  { id: 'inc-010', camId: 'cam-03', location: 'Corridor 2', type: 'Loitering', severity: 'Minor', dateTime: at(4, '13:41:00'), outcome: 'Escalated', confidence: 0.85, description: 'Same person waiting beside the corridor bench for over 20 minutes.', assignedTo: 's2', remark: 'Not a tenant; security asked to check.', duration: '15 min 48 s' },
];
