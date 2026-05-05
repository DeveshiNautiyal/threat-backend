import React, { useState, useEffect } from 'react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { AlertCircle, TrendingUp, Users, Shield, Activity } from 'lucide-react';

export default function SOCDashboard() {
  const [activeTab, setActiveTab] = useState('overview');
  const [metrics, setMetrics] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [userRisks, setUserRisks] = useState([]);
  const [analytics, setAnalytics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedAlert, setSelectedAlert] = useState(null);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 30000); // Refresh every 30s
    return () => clearInterval(interval);
  }, []);

  const fetchData = async () => {
    try {
      const [metricsRes, alertsRes, usersRes, analyticsRes] = await Promise.all([
        fetch('http://localhost:8000/api/metrics').then(r => r.json()),
        fetch('http://localhost:8000/api/alerts?limit=50').then(r => r.json()),
        fetch('http://localhost:8000/api/users/risk?limit=20').then(r => r.json()),
        fetch('http://localhost:8000/api/analytics').then(r => r.json())
      ]);
      
      setMetrics(metricsRes);
      setAlerts(alertsRes);
      setUserRisks(usersRes);
      setAnalytics(analyticsRes);
    } catch (error) {
      console.error('Error fetching data:', error);
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (level) => {
    switch(level) {
      case 'CRITICAL': return 'bg-red-600';
      case 'HIGH': return 'bg-orange-500';
      case 'MEDIUM': return 'bg-yellow-500';
      case 'LOW': return 'bg-blue-500';
      default: return 'bg-gray-500';
    }
  };

  const getRiskTextColor = (level) => {
    switch(level) {
      case 'CRITICAL': return 'text-red-600';
      case 'HIGH': return 'text-orange-500';
      case 'MEDIUM': return 'text-yellow-500';
      case 'LOW': return 'text-blue-500';
      default: return 'text-gray-500';
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-900">
        <div className="text-white text-xl">Loading Insider Threat Detection System...</div>
      </div>
    );
  }

  const RISK_COLORS = {
    CRITICAL: '#dc2626',
    HIGH: '#f97316',
    MEDIUM: '#eab308',
    LOW: '#3b82f6'
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Shield className="w-8 h-8 text-blue-500" />
              <h1 className="text-2xl font-bold">Insider Threat Detection</h1>
            </div>
            <div className="text-gray-400 text-sm">
              Last Updated: {new Date().toLocaleTimeString()}
            </div>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <div className="bg-gray-800 border-b border-gray-700 sticky top-16 z-40">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex gap-0">
            {['overview', 'alerts', 'users', 'analytics'].map(tab => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-6 py-3 font-medium border-b-2 transition-colors ${
                  activeTab === tab
                    ? 'border-blue-500 text-blue-400'
                    : 'border-transparent text-gray-400 hover:text-white'
                }`}
              >
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-8">
        {/* OVERVIEW TAB */}
        {activeTab === 'overview' && metrics && (
          <div className="space-y-8">
            {/* Key Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
              <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-gray-400 text-sm">Total Users</p>
                    <p className="text-3xl font-bold mt-2">{metrics.total_users}</p>
                  </div>
                  <Users className="w-8 h-8 text-blue-500 opacity-50" />
                </div>
              </div>

              <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-gray-400 text-sm">Total Alerts</p>
                    <p className="text-3xl font-bold mt-2">{metrics.total_alerts}</p>
                  </div>
                  <AlertCircle className="w-8 h-8 text-orange-500 opacity-50" />
                </div>
              </div>

              <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-gray-400 text-sm">Critical Alerts</p>
                    <p className="text-3xl font-bold mt-2 text-red-500">{metrics.high_risk_count}</p>
                  </div>
                  <TrendingUp className="w-8 h-8 text-red-500 opacity-50" />
                </div>
              </div>

              <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                <div>
                  <p className="text-gray-400 text-sm">Detection Accuracy</p>
                  <p className="text-3xl font-bold mt-2">{(metrics.detection_accuracy * 100).toFixed(1)}%</p>
                </div>
              </div>

              <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                <div>
                  <p className="text-gray-400 text-sm">Avg Risk Score</p>
                  <p className="text-3xl font-bold mt-2">{metrics.mean_fused_score.toFixed(3)}</p>
                </div>
              </div>
            </div>

            {/* Risk Distribution */}
            {analytics && (
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                  <h3 className="text-lg font-semibold mb-4">Risk Distribution</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={[
                          { name: 'Critical', value: analytics.risk_distribution.CRITICAL },
                          { name: 'High', value: analytics.risk_distribution.HIGH },
                          { name: 'Medium', value: analytics.risk_distribution.MEDIUM },
                          { name: 'Low', value: analytics.risk_distribution.LOW }
                        ]}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                        outerRadius={100}
                        fill="#8884d8"
                        dataKey="value"
                      >
                        <Cell fill={RISK_COLORS.CRITICAL} />
                        <Cell fill={RISK_COLORS.HIGH} />
                        <Cell fill={RISK_COLORS.MEDIUM} />
                        <Cell fill={RISK_COLORS.LOW} />
                      </Pie>
                    </PieChart>
                  </ResponsiveContainer>
                </div>

                <div className="bg-gray-800 rounded-lg p-6 border border-gray-700 lg:col-span-2">
                  <h3 className="text-lg font-semibold mb-4">Alert Timeline</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={analytics.time_series}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                      <XAxis dataKey="date" stroke="#888" />
                      <YAxis stroke="#888" />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                        labelStyle={{ color: '#fff' }}
                      />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="average_risk"
                        stroke="#3b82f6"
                        strokeWidth={2}
                        dot={{ fill: '#3b82f6', r: 3 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ALERTS TAB */}
        {activeTab === 'alerts' && (
          <div className="space-y-4">
            <h2 className="text-2xl font-bold mb-4">Recent Alerts</h2>
            <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-gray-700 border-b border-gray-600">
                    <tr>
                      <th className="px-6 py-3 text-left text-sm font-semibold">User</th>
                      <th className="px-6 py-3 text-left text-sm font-semibold">Date</th>
                      <th className="px-6 py-3 text-left text-sm font-semibold">Baseline Score</th>
                      <th className="px-6 py-3 text-left text-sm font-semibold">Meta Score</th>
                      <th className="px-6 py-3 text-left text-sm font-semibold">Fused Score</th>
                      <th className="px-6 py-3 text-left text-sm font-semibold">Risk Level</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-700">
                    {alerts.map((alert, idx) => (
                      <tr key={idx} className="hover:bg-gray-750 transition-colors">
                        <td className="px-6 py-4 font-mono text-sm">{alert.user}</td>
                        <td className="px-6 py-4 text-sm text-gray-400">{alert.date}</td>
                        <td className="px-6 py-4 text-sm">{alert.baseline_score.toFixed(3)}</td>
                        <td className="px-6 py-4 text-sm">{alert.meta_score.toFixed(3)}</td>
                        <td className="px-6 py-4 font-semibold">{alert.fused_score.toFixed(3)}</td>
                        <td className="px-6 py-4">
                          <span className={`px-3 py-1 rounded-full text-xs font-semibold ${getRiskColor(alert.risk_level)} text-white`}>
                            {alert.risk_level}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* USERS TAB */}
        {activeTab === 'users' && (
          <div className="space-y-4">
            <h2 className="text-2xl font-bold mb-4">User Risk Profiles</h2>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {userRisks.map((user, idx) => (
                <div key={idx} className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                  <div className="flex justify-between items-start mb-4">
                    <div>
                      <h3 className="font-semibold text-lg">{user.user}</h3>
                      <p className="text-gray-400 text-sm">Alert Count: {user.alert_count}</p>
                    </div>
                    <span className={`px-3 py-1 rounded-full text-xs font-semibold ${getRiskColor('HIGH')} text-white`}>
                      {user.risk_trend.toUpperCase()}
                    </span>
                  </div>
                  
                  <div className="space-y-3">
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-gray-400">Average Risk</span>
                        <span className={`font-semibold ${getRiskTextColor('HIGH')}`}>
                          {user.average_risk.toFixed(3)}
                        </span>
                      </div>
                      <div className="w-full bg-gray-700 rounded h-2">
                        <div
                          className="bg-orange-500 h-2 rounded"
                          style={{ width: `${user.average_risk * 100}%` }}
                        ></div>
                      </div>
                    </div>

                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-gray-400">Peak Risk</span>
                        <span className={`font-semibold ${getRiskTextColor('HIGH')}`}>
                          {user.peak_risk.toFixed(3)}
                        </span>
                      </div>
                      <div className="w-full bg-gray-700 rounded h-2">
                        <div
                          className="bg-red-600 h-2 rounded"
                          style={{ width: `${user.peak_risk * 100}%` }}
                        ></div>
                      </div>
                    </div>

                    {user.last_alert_date && (
                      <p className="text-xs text-gray-400">
                        Last Alert: {user.last_alert_date}
                      </p>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ANALYTICS TAB */}
        {activeTab === 'analytics' && analytics && (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold">System Analytics</h2>

            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <h3 className="text-lg font-semibold mb-4">Model Performance</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {Object.entries(analytics.model_performance).map(([key, value]) => (
                  <div key={key} className="bg-gray-700 rounded p-4">
                    <p className="text-gray-400 text-sm capitalize">{key.replace('_', ' ')}</p>
                    <p className="text-2xl font-bold mt-2">{(value * 100).toFixed(1)}%</p>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <h3 className="text-lg font-semibold mb-4">Top Users by Risk</h3>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={analytics.top_users}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                  <XAxis dataKey="user" stroke="#888" />
                  <YAxis stroke="#888" />
                  <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }} />
                  <Legend />
                  <Bar dataKey="average_risk" fill="#f97316" name="Average Risk" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
