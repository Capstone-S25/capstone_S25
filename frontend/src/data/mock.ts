// Demo data used when the incidents API is not reachable.
// The original mock module was not committed with the pages that import it;
// this one follows the shapes in ../types and the ids those pages expect.
import type { Camera, Incident, MonitoringRule, StaffMember } from '../types';

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
  { id: 'cam-01', name: 'Cam 01', location: 'Meeting Room 1', status: 'online', rules: ['rule-01', 'rule-02'] },
  { id: 'cam-02', name: 'Cam 02', location: 'Main Entrance', status: 'online', rules: ['rule-03'] },
  { id: 'cam-03', name: 'Cam 03', location: 'Corridor 2', status: 'online', rules: ['rule-04'] },
  { id: 'cam-04', name: 'Cam 04', location: 'Carpark B2', status: 'online', rules: ['rule-05'] },
  { id: 'cam-05', name: 'Cam 05', location: 'Lift Lobby 1', status: 'online', rules: ['rule-06'] },
  { id: 'cam-06', name: 'Cam 06', location: 'Staircase B', status: 'online', rules: ['rule-07'] },
  { id: 'cam-07', name: 'Cam 07', location: 'Pantry L3', status: 'offline', rules: [], lastSeen: at(0, '18:12:00') },
  { id: 'cam-09', name: 'Cam 09', location: 'Loading Bay', status: 'offline', rules: [], lastSeen: at(1, '09:40:00') },
];

export const MOCK_RULES: MonitoringRule[] = [
  { id: 'rule-01', location: 'Meeting Room 1', cameraIds: ['cam-01'], detectType: 'Smoking', activeFrom: '00:00', activeTo: '23:59', severity: 'Moderate', minConfidence: 0.7, enabled: true },
  { id: 'rule-02', location: 'Meeting Room 1', cameraIds: ['cam-01'], detectType: 'Littering', activeFrom: '08:00', activeTo: '20:00', severity: 'Minor', minConfidence: 0.65, enabled: true },
  { id: 'rule-03', location: 'Main Entrance', cameraIds: ['cam-02'], detectType: 'Loitering', activeFrom: '22:00', activeTo: '06:00', severity: 'Minor', minConfidence: 0.7, enabled: true },
  { id: 'rule-04', location: 'Corridor 2', cameraIds: ['cam-03'], detectType: 'Unattended item', activeFrom: '00:00', activeTo: '23:59', severity: 'Critical', minConfidence: 0.75, enabled: true },
  { id: 'rule-05', location: 'Carpark B2', cameraIds: ['cam-04'], detectType: 'Smoking', activeFrom: '22:00', activeTo: '06:00', severity: 'Critical', minConfidence: 0.7, enabled: true },
  { id: 'rule-06', location: 'Lift Lobby 1', cameraIds: ['cam-05'], detectType: 'Unattended item', activeFrom: '00:00', activeTo: '23:59', severity: 'Moderate', minConfidence: 0.7, enabled: true },
  { id: 'rule-07', location: 'Staircase B', cameraIds: ['cam-06'], detectType: 'Smoking', activeFrom: '00:00', activeTo: '23:59', severity: 'Moderate', minConfidence: 0.7, enabled: false },
];

export const MOCK_INCIDENTS: Incident[] = [
  { id: 'inc-001', camId: 'cam-04', location: 'Carpark B2', type: 'Smoking', severity: 'Critical', dateTime: at(0, '23:18:06'), outcome: 'Pending', confidence: 0.92, description: 'Person smoking a cigarette next to the staircase door, ash dropped on the floor.', duration: '3 min 32 s' },
  { id: 'inc-002', camId: 'cam-01', location: 'Meeting Room 1', type: 'Unattended item', severity: 'Moderate', dateTime: at(0, '10:15:00'), outcome: 'Pending', confidence: 0.81, description: 'A backpack left on a chair after the room emptied.', duration: '12 min 05 s' },
  { id: 'inc-003', camId: 'cam-03', location: 'Corridor 2', type: 'Loitering', severity: 'Minor', dateTime: at(0, '11:02:00'), outcome: 'Pending', confidence: 0.74, description: 'One person standing by the fire exit without moving on.', duration: '6 min 40 s' },
  { id: 'inc-004', camId: 'cam-05', location: 'Lift Lobby 1', type: 'Littering', severity: 'Minor', dateTime: at(0, '20:37:00'), outcome: 'Pending', confidence: 0.68, description: 'A drink cup left on the floor beside the lift.', duration: '0 min 18 s' },
  { id: 'inc-005', camId: 'cam-06', location: 'Staircase B', type: 'Smoking', severity: 'Moderate', dateTime: at(1, '14:22:00'), outcome: 'Escalated', confidence: 0.88, description: 'Two people smoking on the landing.', assignedTo: 's1', remark: 'Repeat location this week.', duration: '4 min 10 s' },
  { id: 'inc-006', camId: 'cam-02', location: 'Main Entrance', type: 'Loitering', severity: 'Minor', dateTime: at(1, '02:05:00'), outcome: 'Dismissed', confidence: 0.61, description: 'Person waiting near the entrance.', assignedTo: 's2', remark: 'Delivery rider waiting for pickup.', duration: '7 min 02 s' },
  { id: 'inc-007', camId: 'cam-04', location: 'Carpark B2', type: 'Unattended item', severity: 'Critical', dateTime: at(2, '17:48:00'), outcome: 'Accepted', confidence: 0.9, description: 'A box left between two parked cars.', assignedTo: 's1', duration: '21 min 30 s' },
  { id: 'inc-008', camId: 'cam-05', location: 'Lift Lobby 1', type: 'Littering', severity: 'Minor', dateTime: at(2, '09:12:00'), outcome: 'Accepted', confidence: 0.79, description: 'Food wrapper dropped while leaving the lift.', assignedTo: 's2', duration: '0 min 12 s' },
  { id: 'inc-009', camId: 'cam-01', location: 'Meeting Room 1', type: 'Smoking', severity: 'Moderate', dateTime: at(3, '19:30:00'), outcome: 'Dismissed', confidence: 0.66, description: 'Hand raised to the mouth repeatedly.', assignedTo: 's1', remark: 'Drinking from a bottle, not smoking.', duration: '1 min 05 s' },
  { id: 'inc-010', camId: 'cam-03', location: 'Corridor 2', type: 'Unattended item', severity: 'Critical', dateTime: at(4, '13:41:00'), outcome: 'Escalated', confidence: 0.85, description: 'Suitcase left beside the corridor bench.', assignedTo: 's2', remark: 'Owner not seen returning.', duration: '15 min 48 s' },
];
