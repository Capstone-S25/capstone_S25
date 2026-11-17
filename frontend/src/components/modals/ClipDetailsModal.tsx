import { Incident } from '../../types/index';
import ClipDetailDrawer from '../drawers/ClipDetailDrawer';

interface ClipDetailsModalProps {
  isOpen: boolean;
  onClose: () => void;
  incident: Incident | null;
}

/** Kept for existing call sites — clip details now share the side drawer. */
export default function ClipDetailsModal({ isOpen, onClose, incident }: ClipDetailsModalProps) {
  return <ClipDetailDrawer inc={isOpen ? incident : null} onClose={onClose} />;
}
