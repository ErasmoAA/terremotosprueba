import React, { useState, useEffect, useRef } from 'react';
import { AlertTriangle, Globe, Brain, TrendingUp, MapPin, Activity, Database, Zap } from 'lucide-react';
import * as tf from 'tensorflow';

const QuakeWatcherApp = () => {
  const [earthquakeData, setEarthquakeData] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [model, setModel] = useState(null);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [selectedRegion, setSelectedRegion] = useState('user');
  const [userLocation, setUserLocation] = useState(null);
  const mapRef = useRef(null);

  // Regiones predefinidas por si la geolocalización falla
  const regions = {
    california: { name: 'California', lat: 36.7783, lng: -119.4179, bbox: '32,-125,42,-114' },
    alaska: { name: 'Alaska', lat: 64.0685, lng: -152.2782, bbox: '54,-179,71,-129' },
    yellowstone: { name: 'Yellowstone', lat: 44.6, lng: -110.5, bbox: '44,-111,45,-109' },
    newmadrid: { name: 'New Madrid', lat: 36.5861, lng: -89.5889, bbox: '35,-91,38,-88' }
  };

  // Solicitar geolocalización del usuario
  useEffect(() => {
    if (!userLocation && navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          setUserLocation({ lat: pos.coords.latitude, lng: pos.coords.longitude });
        },
        (err) => {
          console.error('Geolocation error:', err);
          setUserLocation(null);
        }
      );
    }
  }, [userLocation]);

  // Cargar datos reales del USGS según región o ubicación
  const loadUSGSData = async (regionKey, location) => {
    try {
      setLoading(true);
      const endTime = new Date().toISOString();
      const startTime = new Date(Date.now() - 365 * 24 * 60 * 60 * 1000).toISOString();
      let url = '';

      if (regionKey === 'user' && location) {
        // usar radio grande para incluir eventos que puedan afectar al usuario
        url = `https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=${startTime}&endtime=${endTime}&latitude=${location.lat}&longitude=${location.lng}&maxradiuskm=1000&limit=2000&minmagnitude=2.5&orderby=time`;
      } else {
        const bbox = regions[regionKey].bbox;
        url = `https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=${startTime}&endtime=${endTime}&minmagnitude=2.5&orderby=time&bbox=${bbox}&limit=2000`;
      }

      const response = await fetch(url);
      const data = await response.json();

      const filteredData = data.features.map((feature) => ({
        id: feature.id,
        magnitude: feature.properties.mag,
        place: feature.properties.place,
        time: new Date(feature.properties.time),
        lat: feature.geometry.coordinates[1],
        lng: feature.geometry.coordinates[0],
        depth: feature.geometry.coordinates[2]
      })).slice(0, 2000);

      setEarthquakeData(filteredData);
      return filteredData;
    } catch (error) {
      console.error('Error loading USGS data:', error);
      return [];
    } finally {
      setLoading(false);
    }
  };

  // Crear y entrenar red neuronal
  const createNeuralNetwork = async (data) => {
    try {
      setTrainingProgress(0);
      const features = data.map((eq) => [
        eq.magnitude,
        eq.depth,
        eq.lat,
        eq.lng,
        eq.time.getTime() / 1000000000,
        Math.sin(eq.time.getMonth() * Math.PI / 6),
        Math.cos(eq.time.getMonth() * Math.PI / 6)
      ]);

      const labels = data.map((eq, index) => {
        const futureEqs = data.slice(0, index).filter(
          (futureEq) =>
            futureEq.time > eq.time &&
            futureEq.time < new Date(eq.time.getTime() + 30 * 24 * 60 * 60 * 1000)
        );
        return futureEqs.length > 0 ? 1 : 0;
      });

      const xs = tf.tensor2d(features);
      const ys = tf.tensor2d(labels.map((l) => [l]));

      const neuralModel = tf.sequential({
        layers: [
          tf.layers.dense({ inputShape: [7], units: 64, activation: 'relu' }),
          tf.layers.dropout({ rate: 0.2 }),
          tf.layers.dense({ units: 32, activation: 'relu' }),
          tf.layers.dropout({ rate: 0.2 }),
          tf.layers.dense({ units: 16, activation: 'relu' }),
          tf.layers.dense({ units: 1, activation: 'sigmoid' })
        ]
      });

      neuralModel.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
      });

      await neuralModel.fit(xs, ys, {
        epochs: 50,
        batchSize: 32,
        validationSplit: 0.2,
        callbacks: {
          onEpochEnd: (epoch) => {
            setTrainingProgress(((epoch + 1) / 50) * 100);
          }
        }
      });

      setModel(neuralModel);
      const currentTime = new Date();
      const predictions = await generatePredictions(neuralModel, currentTime, locationForRegion(selectedRegion));
      setPredictions(predictions);
      xs.dispose();
      ys.dispose();
    } catch (error) {
      console.error('Error creating neural network:', error);
      setPredictions([]);
    }
  };

  const locationForRegion = (regionKey) => {
    if (regionKey === 'user' && userLocation) return { lat: userLocation.lat, lng: userLocation.lng };
    if (regions[regionKey]) return { lat: regions[regionKey].lat, lng: regions[regionKey].lng };
    return { lat: 0, lng: 0 };
  };

  // Generar predicciones con el modelo
  const generatePredictions = async (model, currentTime, loc) => {
    const predictions = [];
    const region = loc;
    for (let i = 0; i < 10; i++) {
      const lat = region.lat + (Math.random() - 0.5) * 2;
      const lng = region.lng + (Math.random() - 0.5) * 2;
      const depth = Math.random() * 30;
      const magnitude = 3 + Math.random() * 3;
      const features = tf.tensor2d([
        [
          magnitude,
          depth,
          lat,
          lng,
          currentTime.getTime() / 1000000000,
          Math.sin(currentTime.getMonth() * Math.PI / 6),
          Math.cos(currentTime.getMonth() * Math.PI / 6)
        ]
      ]);
      const prediction = await model.predict(features).data();
      features.dispose();
      predictions.push({
        id: i,
        lat,
        lng,
        probability: prediction[0] * 100,
        magnitude,
        depth,
        region: selectedRegion === 'user' ? 'Mi ubicación' : regions[selectedRegion].name,
        timeframe: `${7 + Math.floor(Math.random() * 23)} días`
      });
    }
    return predictions.sort((a, b) => b.probability - a.probability);
  };

  // Efecto principal
  useEffect(() => {
    const initializeApp = async () => {
      const regionKey = selectedRegion === 'user' && !userLocation ? 'california' : selectedRegion;
      const location = selectedRegion === 'user' ? userLocation : null;
      const data = await loadUSGSData(regionKey, location);
      if (data.length > 10) {
        await createNeuralNetwork(data);
      }
    };
    initializeApp();
  }, [selectedRegion, userLocation]);

  const getRiskLevel = () => {
    if (predictions.length === 0) return 'low';
    const avgProbability = predictions.reduce((sum, p) => sum + p.probability, 0) / predictions.length;
    if (avgProbability > 60) return 'high';
    if (avgProbability > 30) return 'medium';
    return 'low';
  };

  const currentRiskLevel = getRiskLevel();

  const mapLink = (lat, lng) => `https://www.google.com/maps?q=${lat},${lng}`;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-800 text-white">
      <header className="bg-black/20 backdrop-blur-sm border-b border-white/10">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-red-500/20 rounded-lg">
                <Globe className="h-8 w-8 text-red-400" />
              </div>
              <div>
                <h1 className="text-2xl font-bold">QuakeWatcher AI</h1>
                <p className="text-sm text-gray-300">Predicción de Terremotos</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Database className="h-5 w-5 text-green-400" />
                <span className="text-sm">Datos USGS</span>
              </div>
              <div className="flex items-center space-x-2">
                <Brain className="h-5 w-5 text-blue-400" />
                <span className="text-sm">Red Neuronal Activa</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-6 py-6">
        <div className="flex flex-wrap gap-3 mb-6">
          {userLocation && (
            <button
              onClick={() => setSelectedRegion('user')}
              className={`px-4 py-2 rounded-lg font-medium transition-all ${
                selectedRegion === 'user'
                  ? 'bg-blue-500 text-white shadow-lg'
                  : 'bg-white/10 text-gray-300 hover:bg-white/20'
              }`}
            >
              Mi ubicación
            </button>
          )}
          {Object.entries(regions).map(([key, region]) => (
            <button
              key={key}
              onClick={() => setSelectedRegion(key)}
              className={`px-4 py-2 rounded-lg font-medium transition-all ${
                selectedRegion === key
                  ? 'bg-blue-500 text-white shadow-lg'
                  : 'bg-white/10 text-gray-300 hover:bg-white/20'
              }`}
            >
              {region.name}
            </button>
          ))}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-black/30 backdrop-blur-sm rounded-xl p-6 border border-white/10">
            <div className="flex items-center space-x-3 mb-2">
              <AlertTriangle className={`h-6 w-6 ${
                currentRiskLevel === 'high'
                  ? 'text-red-400'
                  : currentRiskLevel === 'medium'
                  ? 'text-yellow-400'
                  : 'text-green-400'
              }`} />
              <h3 className="font-semibold">Nivel de Riesgo</h3>
            </div>
            <p
              className={`text-2xl font-bold ${
                currentRiskLevel === 'high'
                  ? 'text-red-400'
                  : currentRiskLevel === 'medium'
                  ? 'text-yellow-400'
                  : 'text-green-400'
              }`}
            >
              {currentRiskLevel === 'high'
                ? 'ALTO'
                : currentRiskLevel === 'medium'
                ? 'MEDIO'
                : 'BAJO'}
            </p>
          </div>

          <div className="bg-black/30 backdrop-blur-sm rounded-xl p-6 border border-white/10">
            <div className="flex items-center space-x-3 mb-2">
              <Activity className="h-6 w-6 text-blue-400" />
              <h3 className="font-semibold">Datos Analizados</h3>
            </div>
            <p className="text-2xl font-bold text-blue-400">{earthquakeData.length}</p>
            <p className="text-sm text-gray-400">Terremotos históricos</p>
          </div>

          <div className="bg-black/30 backdrop-blur-sm rounded-xl p-6 border border-white/10">
            <div className="flex items-center space-x-3 mb-2">
              <Brain className="h-6 w-6 text-purple-400" />
              <h3 className="font-semibold">IA Entrenada</h3>
            </div>
            <p className="text-2xl font-bold text-purple-400">{model ? '✓' : loading ? '...' : '✗'}</p>
            <p className="text-sm text-gray-400">
              {trainingProgress > 0 && trainingProgress < 100
                ? `${trainingProgress.toFixed(0)}%`
                : model
                ? 'Red activa'
                : 'Entrenando...'}
            </p>
          </div>

          <div className="bg-black/30 backdrop-blur-sm rounded-xl p-6 border border-white/10">
            <div className="flex items-center space-x-3 mb-2">
              <TrendingUp className="h-6 w-6 text-green-400" />
              <h3 className="font-semibold">Predicciones</h3>
            </div>
            <p className="text-2xl font-bold text-green-400">{predictions.length}</p>
            <p className="text-sm text-gray-400">Próximos 30 días</p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="bg-black/30 backdrop-blur-sm rounded-xl p-6 border border-white/10">
            <h2 className="text-xl font-bold mb-4 flex items-center">
              <Zap className="h-5 w-5 mr-2 text-yellow-400" />
              Predicciones Sísmicas
            </h2>
            {loading ? (
              <div className="text-center py-8">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-400 mx-auto"></div>
                <p className="mt-4 text-gray-400">Cargando datos del USGS...</p>
              </div>
            ) : (
              <div className="space-y-4 max-h-96 overflow-y-auto">
                {predictions.map((prediction) => (
                  <a
                    key={prediction.id}
                    href={mapLink(prediction.lat, prediction.lng)}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="block bg-white/5 rounded-lg p-4 border border-white/10 hover:bg-white/10"
                  >
                    <div className="flex justify-between items-start mb-2">
                      <div className="flex items-center space-x-2">
                        <MapPin className="h-4 w-4 text-red-400" />
                        <span className="font-medium">{prediction.region}</span>
                      </div>
                      <div
                        className={`px-2 py-1 rounded text-xs font-medium ${
                          prediction.probability > 60
                            ? 'bg-red-500/20 text-red-300'
                            : prediction.probability > 30
                            ? 'bg-yellow-500/20 text-yellow-300'
                            : 'bg-green-500/20 text-green-300'
                        }`}
                      >
                        {prediction.probability.toFixed(1)}% probabilidad
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <p className="text-gray-400">Magnitud estimada</p>
                        <p className="font-medium">{prediction.magnitude.toFixed(1)}</p>
                      </div>
                      <div>
                        <p className="text-gray-400">Profundidad</p>
                        <p className="font-medium">{prediction.depth.toFixed(1)} km</p>
                      </div>
                      <div>
                        <p className="text-gray-400">Coordenadas</p>
                        <p className="font-medium">
                          {prediction.lat.toFixed(2)}, {prediction.lng.toFixed(2)}
                        </p>
                      </div>
                      <div>
                        <p className="text-gray-400">Tiempo estimado</p>
                        <p className="font-medium">{prediction.timeframe}</p>
                      </div>
                    </div>
                  </a>
                ))}
              </div>
            )}
          </div>

          <div className="bg-black/30 backdrop-blur-sm rounded-xl p-6 border border-white/10">
            <h2 className="text-xl font-bold mb-4 flex items-center">
              <Database className="h-5 w-5 mr-2 text-green-400" />
              Datos Históricos USGS
            </h2>
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {earthquakeData.slice(0, 10).map((earthquake) => (
                <div key={earthquake.id} className="bg-white/5 rounded-lg p-3 border border-white/10">
                  <div className="flex justify-between items-start mb-2">
                    <div className="flex items-center space-x-2">
                      <div
                        className={`w-3 h-3 rounded-full ${
                          earthquake.magnitude >= 5
                            ? 'bg-red-400'
                            : earthquake.magnitude >= 4
                            ? 'bg-yellow-400'
                            : 'bg-green-400'
                        }`}
                      ></div>
                      <span className="font-medium">M {earthquake.magnitude.toFixed(1)}</span>
                    </div>
                    <span className="text-xs text-gray-400">{earthquake.time.toLocaleDateString()}</span>
                  </div>
                  <p className="text-sm text-gray-300 mb-1">{earthquake.place}</p>
                  <p className="text-xs text-gray-400">
                    {earthquake.lat.toFixed(2)}, {earthquake.lng.toFixed(2)} | {earthquake.depth.toFixed(1)} km profundidad
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="mt-8 bg-yellow-500/10 border border-yellow-500/20 rounded-xl p-6">
          <div className="flex items-start space-x-3">
            <AlertTriangle className="h-6 w-6 text-yellow-400 mt-0.5" />
            <div>
              <h3 className="font-semibold text-yellow-400 mb-2">Advertencia Científica</h3>
              <p className="text-sm text-gray-300 leading-relaxed">
                Esta aplicación utiliza datos reales del USGS y técnicas de machine learning para generar
                predicciones probabilísticas. Las predicciones de terremotos siguen siendo uno de los mayores
                desafíos científicos. Los resultados mostrados son experimentales y no deben usarse para tomar
                decisiones de seguridad. Consulte siempre fuentes oficiales como el USGS para información sísmica
                autorizada.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default QuakeWatcherApp;
