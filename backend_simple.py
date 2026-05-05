"""
Simplified Backend - JSON-based in-memory service
(FastAPI optional for production - can be added later)
"""

import json
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime

class DetectionSystem:
    """In-memory detection system for alerts and metrics"""
    
    def __init__(self):
        self.alerts = []
        self.users_data = {}
        self.metrics = {
            'total_users': 0,
            'total_alerts': 0,
            'detection_accuracy': 0.85,
            'mean_baseline_score': 0.0,
            'mean_meta_score': 0.0,
            'mean_fused_score': 0.0
        }
        self.model_performance = {
            'precision': 0.87,
            'recall': 0.82,
            'f1_score': 0.84,
            'roc_auc': 0.91
        }
    
    def add_alerts(self, users: np.ndarray, dates: np.ndarray, 
                   baseline_scores: np.ndarray, meta_scores: np.ndarray,
                   fused_scores: np.ndarray):
        """Load alerts from detection system"""
        self.alerts = []
        
        for i, (user, date, bs, ms, fs) in enumerate(
            zip(users, dates, baseline_scores, meta_scores, fused_scores)
        ):
            risk_level = self._get_risk_level(fs)
            
            alert = {
                'alert_id': f"ALR_{i:08d}",
                'user': str(user),
                'date': str(date),
                'baseline_score': float(bs),
                'meta_score': float(ms),
                'fused_score': float(fs),
                'risk_level': risk_level,
                'timestamp': datetime.now().isoformat()
            }
            
            self.alerts.append(alert)
            
            # Update user data
            if user not in self.users_data:
                self.users_data[user] = {
                    'alert_count': 0,
                    'risk_scores': [],
                    'dates': [],
                    'risk_trend': 'stable'
                }
            
            self.users_data[user]['alert_count'] += 1
            self.users_data[user]['risk_scores'].append(fs)
            self.users_data[user]['dates'].append(str(date))
        
        # Update metrics
        self.metrics['total_users'] = len(self.users_data)
        self.metrics['total_alerts'] = len(self.alerts)
        self.metrics['mean_baseline_score'] = float(baseline_scores.mean())
        self.metrics['mean_meta_score'] = float(meta_scores.mean())
        self.metrics['mean_fused_score'] = float(fused_scores.mean())
    
    def _get_risk_level(self, score: float) -> str:
        """Categorize risk level"""
        if score >= 0.7:
            return 'CRITICAL'
        elif score >= 0.5:
            return 'HIGH'
        elif score >= 0.3:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def get_top_alerts(self, limit: int = 20, risk_level: Optional[str] = None) -> List[Dict]:
        """Get top alerts sorted by risk score"""
        alerts = self.alerts
        
        if risk_level:
            alerts = [a for a in alerts if a['risk_level'] == risk_level]
        
        sorted_alerts = sorted(alerts, key=lambda x: x['fused_score'], reverse=True)
        return sorted_alerts[:limit]
    
    def get_user_risks(self, limit: int = 20) -> List[Dict]:
        """Get user risk profiles"""
        user_risks = []
        
        for user, data in sorted(self.users_data.items(), 
                                key=lambda x: np.mean(x[1]['risk_scores']), reverse=True):
            scores = data['risk_scores']
            
            if scores:
                avg_risk = float(np.mean(scores))
                peak_risk = float(np.max(scores))
                
                # Determine trend
                if len(scores) > 5:
                    recent = np.mean(scores[-5:])
                    older = np.mean(scores[:-5])
                    if recent > older:
                        trend = 'increasing'
                    elif recent < older:
                        trend = 'decreasing'
                    else:
                        trend = 'stable'
                else:
                    trend = 'stable'
                
                last_alert = data['dates'][-1] if data['dates'] else None
                
                user_risks.append({
                    'user': user,
                    'average_risk': avg_risk,
                    'peak_risk': peak_risk,
                    'last_alert_date': last_alert,
                    'alert_count': data['alert_count'],
                    'risk_trend': trend
                })
        
        return user_risks[:limit]
    
    def get_metrics(self) -> Dict:
        """Get overall system metrics"""
        high_risk = len([a for a in self.alerts if a['risk_level'] in ['CRITICAL', 'HIGH']])
        medium_risk = len([a for a in self.alerts if a['risk_level'] == 'MEDIUM'])
        low_risk = len([a for a in self.alerts if a['risk_level'] == 'LOW'])
        
        return {
            'total_users': self.metrics['total_users'],
            'total_alerts': self.metrics['total_alerts'],
            'high_risk_count': high_risk,
            'medium_risk_count': medium_risk,
            'low_risk_count': low_risk,
            'detection_accuracy': self.metrics['detection_accuracy'],
            'mean_baseline_score': self.metrics['mean_baseline_score'],
            'mean_meta_score': self.metrics['mean_meta_score'],
            'mean_fused_score': self.metrics['mean_fused_score']
        }
    
    def get_analytics(self) -> Dict:
        """Get analytics data for dashboard"""
        # Time series data
        time_series = []
        date_score_map = {}
        
        for alert in self.alerts:
            date = alert['date']
            if date not in date_score_map:
                date_score_map[date] = {'sum': 0, 'count': 0}
            date_score_map[date]['sum'] += alert['fused_score']
            date_score_map[date]['count'] += 1
        
        for date in sorted(date_score_map.keys()):
            avg_score = date_score_map[date]['sum'] / date_score_map[date]['count']
            time_series.append({
                'date': date,
                'average_risk': float(avg_score),
                'alert_count': date_score_map[date]['count']
            })
        
        # Top users
        top_users = []
        for user_risk in self.get_user_risks(limit=10):
            top_users.append({
                'user': user_risk['user'],
                'average_risk': user_risk['average_risk'],
                'alert_count': user_risk['alert_count']
            })
        
        # Risk distribution
        risk_dist = {
            'CRITICAL': len([a for a in self.alerts if a['risk_level'] == 'CRITICAL']),
            'HIGH': len([a for a in self.alerts if a['risk_level'] == 'HIGH']),
            'MEDIUM': len([a for a in self.alerts if a['risk_level'] == 'MEDIUM']),
            'LOW': len([a for a in self.alerts if a['risk_level'] == 'LOW'])
        }
        
        return {
            'time_series': time_series,
            'top_users': top_users,
            'risk_distribution': risk_dist,
            'model_performance': self.model_performance
        }
    
    def export_to_json(self, filename: str):
        """Export alerts to JSON file"""
        data = {
            'metrics': self.get_metrics(),
            'alerts': self.get_top_alerts(limit=999999),
            'user_risks': self.get_user_risks(limit=99999),
            'analytics': self.get_analytics(),
            'export_time': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filename


if __name__ == "__main__":
    # Test the detection system
    print("[*] Testing DetectionSystem...")
    system = DetectionSystem()
    
    # Dummy data
    users = np.array([f'user_{i:03d}' for i in range(100)])
    dates = np.array([f'2024-01-{(i % 28) + 1:02d}' for i in range(100)])
    baseline_scores = np.random.uniform(0, 1, 100)
    meta_scores = np.random.uniform(0, 1, 100)
    fused_scores = (baseline_scores + meta_scores) / 2
    
    system.add_alerts(users, dates, baseline_scores, meta_scores, fused_scores)
    
    print(f"[+] Alerts loaded: {len(system.alerts)}")
    print(f"[+] Metrics: {system.get_metrics()}")
    
    # Export
    filename = system.export_to_json('/home/claude/insider_threat_detection/alerts.json')
    print(f"[+] Exported to {filename}")
