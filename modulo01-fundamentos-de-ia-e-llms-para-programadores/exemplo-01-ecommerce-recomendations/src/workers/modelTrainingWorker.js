import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';

console.log('Model training worker initialized');
let _globalCtx = {};

const normalize = (value, min, max) => (value - min) / ((max - min) || 1);

function makeContext(catalog, users) {
    // Validate inputs
    if (!users || !Array.isArray(users)) {
        console.error('Invalid users data:', users);
        throw new Error('Users must be a non-empty array');
    }
    
    if (!catalog || !Array.isArray(catalog)) {
        console.error('Invalid catalog data:', catalog);
        throw new Error('Catalog must be a non-empty array');
    }

    const ages = users.map(u => u.age);
    const prices = catalog.map(p => p.price);

    const minAge = Math.min(...ages);
    const maxAge = Math.max(...ages);

    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    
    const colors = [...new Set(catalog.map(p => p.color))];
    const categories = [...new Set(catalog.map(p => p.category))];

    const colorIndex = Object.entries(
        colors.map((color, index) => {
            return [color, index]
        }))

    const categoriesIndex = Object.entries(
        categories.map((category, index) => {
            return [category, index]
        })) 

    // Computar a média de idade dos compradores por produto
    // (ajuda a personalizar as recomendações com base na idade)
    const midAge = (minAge + maxAge) / 2;
    const ageSums = {};
    const ageCounts = {};

    users.forEach(user => {
        if (user.purchasedProducts && Array.isArray(user.purchasedProducts)) {
            user.purchasedProducts.forEach(p => {
                ageSums[p] = (ageSums[p] || 0) + user.age;
                ageCounts[p] = (ageCounts[p] || 0) + 1; 
            })
        }
    })

    const productAvgAgeNorm = Object.fromEntries(
        catalog.map(p => {
            const avg = ageCounts[p.name] ? ageSums[p.name] / ageCounts[p.name] : midAge;
            return [p.name, normalize(avg, minAge, maxAge)];
        })
    );
    
    return {
        catalog,
        users,
        colorIndex,
        categoriesIndex,
        minAge,
        maxAge,
        minPrice,
        maxPrice,
        numCategories: categories.length,
        numColors: colors.length,
        dimentions: 2 + categories.length + colors.length, // idade + preço + categorias + cores
    };
}

async function trainModel({ users }) {
    try {
        console.log('Training model with users:', users)
        postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } });
        const catalog = await (await fetch('/data/products.json')).json();
        
        const context = makeContext(catalog, users)

        postMessage({
            type: workerEvents.trainingLog,
            epoch: 1,
            loss: 1,
            accuracy: 1
        });

        setTimeout(() => {
            postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
            postMessage({ type: workerEvents.trainingComplete });
        }, 1000);
    } catch (error) {
        console.error('Error training model:', error);
        postMessage({ 
            type: workerEvents.trainingError, 
            error: error.message 
        });
    }
}
function recommend(user, ctx) {
    console.log('will recommend for user:', user)
    // postMessage({
    //     type: workerEvents.recommend,
    //     user,
    //     recommendations: []
    // });
}


const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: d => recommend(d.user, _globalCtx),
};

self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
