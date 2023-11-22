import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from PIL import Image


st.set_page_config(layout="wide")


with st.sidebar:
    st.image("logo-unison.png", width=300)  # Adjust path and width as needed
    st.markdown("""
    Universidad de Sonora   
    Programa de Doctorado en Ciencias Matemáticas  
    ---------------
        
    **Elaborado por:** María Elena Martínez Manzanares  
    **Nombre de tesis:** Modelos de Control semi-Markovianos para sistemas de grandes poblaciones bajo un enfoque de campo medio   
    **Director de tesis:** Dr. Jesús Adolfo Minjárez Sosa  
    **Fecha de última modificación:** 12-11-2023
    """)

#region Cabecera

st.markdown("""# Simulación de la evolución de proporciones de objetos en una cantidad finita de clases bajo un enfoque de campo medio""")


st.markdown("""
            
            En este tablero podrá simular la evolución de las proporciones de objetos en una cantidad finita de clases. 
            Cuando usted accede por primera vez, la simulación es generada con valores por defecto, pero es posible
            personalizar esto. Para poder realizar su simulación con los valores de su preferencia, es necesario indicar lo siguiente:

            - Matriz de transición entre clases, la cual debe subir como un csv.
            - Número de iteraciones (o etapas) que desea simular. 
            - Tamaño de la población $N$.
            - Proporción (distribución) de objetos en cada clase cuando comienza a correr el sistema (tiempo cero).
            """)

with st.expander("Teoría detras del tablero"):
    st.markdown(
        """
        Dado un sistema controlado que se compone de una cantidad finita de clases $S=\{1,2,...,s\}$ y $N$ objetos que se distribuyen a lo largo del conjunto
        de $S$ clases, es posible estudiar la evolución del sistema en las etapas por medio de la proporción de objetos de cada clase.  

        Particularmente, si denotamos a $\{M^N(k)\}_{k\in\mathbb{N}_0}$ los vectores de proporciones de los objetos en las $S$ clases durante las épocas
        $k=0,1,2...$, es posible demostrar que el proceso $\{M^N(k)\}_{k\in\mathbb{N}_0}$ es una cadena de Markov. Este resultado nos permite utilizar 
        métodos de simulación de Monte Carlo para obtener una ecuación en diferencias estocásticas que define la evolución del proceso. 
        Específicamente obtenemos 
        """)
    st.latex(r"""M^N(k+1)=H^N(M^N(k),a_k,w_k), k\in\mathbb{N}_0""")
    st.markdown("""
        donde $\{w_k\}$ es una sucesión de variables aleatorias i.i.d. en $\mathbb{R}^N$ con distribución theta.  

        Para lograr esto, comencemos considerando $a\in A$, $i,j\in S$, y la siguiente partición del $[0,1]$
        $$
        \Delta_{ij}(a):=[\phi_{i(j-1)}(a),\phi_{ij}(a)]\subseteq [0,1], 
        $$
        donde 
        $$
        \phi_{i0}(a)\equiv 0,\ \ \phi_{ij}(a):=\sum_{l=1}^{j}K_{il}(a),i,j\in S.
        $$
        Para cada $i\in S$, y $k\in\mathbb{N}_0$, definimos
        $$
        w^{i}(k):=(w_{1}^{i}(k),...,w_{NM_{i}^{N}(k)}^{i}(k));
        $$  
        $$
        w_{k}:=\{w^{i}(k)\}_{i\in S}.
        $$
        donde $\{w_n^i(k)\}$  es una familia de variables aleatorias uniformemente distribuidas en $[0,1]$ con $n\in\{1,2,...,N\}$. Considerando que $\sum_{j=1}^{s} NM_j^N(k)=N$, se tiene que $w_k\in[0,1]^N$. Entonces
        """)
    st.latex(r"""M_{j}^{N}(k+1)= \frac{1}{N}\sum_{i=1}^{s}\sum_{n=1}^{NM_{i}^{N}(k)}1_{\Delta_{ij}(a_{k})}(w_{n}^{i}(k)). \quad (1)""")
    st.markdown("""
        Finalmente, se define la función $H^N$ como
        """)
    st.latex(r"""H^{N}(m,a,w):=\{H_{i}^{N}(m,a,w)\}_{i\in S},\quad(m,a,w)\in\mathbb{P}_{N}(S) \times A \times [0,1]^{N},""")
    st.markdown("""
        donde
        """)
    st.latex(r"""H_{j}^{N}(m,a,w)=\frac{1}{N}\sum_{i=1}^{s}\sum_{n=1}^{Nm_{i}}1_{\Delta _{ij}(a)}(w_{n}^{i}(k)),""")
    st.markdown("""    
        y $m=\{m_{i}\}_{i\in S}$, $k\in \mathbb{N}_0$.  

        En este tablero, por simpleza consideraremos evoluciones no controladas, es decir, de la forma $M^N(k+1)=H^N(M^N(k),w_k)$ con $K_{ij}(a)\equiv K_{ij}$.  

        Una característica que puede ser demostrada y cumplen las proporciones determinadas por (1), es que conforme el número de objetos $N$ crece, 
        la distribución (o también llamada configuración) de los objetos converge a una configuración límite. Este resultado se le conoce actualmente 
        en control estocástico como *convergencia de campo medio*.  

        Por medio de este tablero, en la gráfica izquierda es posible ver la evolución de la configuración de objetos simulada por medio de (1). 
        Cuando $N$ es suficientemente grande, los valores que se observan gráficados puede ser considerados como los valores del campo medio.  

        Adicionalmente, en la gráfica de la derecha es posible ver la convergencia hacia el campo medio cuando se mantiene fija la etapa, pero la población aumenta.  
        
        En este tablero, dado que python es cero indexado, la enumeración en $S$ comienza en $0$, es decir, $S=\{0,1,2,...,s-1\}$.
    """
    )
#endregion

#region Inicialización default del tablero e inputs de usuario

# Define default transition matrix
default_transition_matrix = np.array([
    [0.8, 0.15, 0.05],
    [0.3, 0.4, 0.3],
    [0.1, 0.3, 0.6]
])
num_states = default_transition_matrix.shape[0]

# Define default proportions for the population distribution
default_proportions_str = "0.5,0.25,0.25"
default_proportions = np.array([float(x) for x in default_proportions_str.split(',')])
default_population_size = 100
 
# Set up the file uploader
uploaded_file = st.file_uploader("Cargue la matriz de transición como CSV", type=["csv"])

# Initialize with default values if no file is uploaded
transition_matrix = default_transition_matrix if uploaded_file is None else None

# User input for the number of iterations
num_iterations = st.number_input('Número de iteraciones', value=10, min_value=1)

# User input for the total population size
population_size = st.number_input('Tamaño de la población N', min_value=1, value=default_population_size)

# User input for the initial population distribution
proportions_str = st.text_input(
    'Ingrese la distribución de la población en las clases (deben sumar 1)',
    value=default_proportions_str
)
proportions = np.array([float(x) for x in proportions_str.split(',')])

# Validate that the proportions sum up to 1
if not np.isclose(proportions.sum(), 1.0):
    st.error('Las proporciones deben sumar 1.')
    st.stop()

submit_button = st.button('Iniciar simulación')

# Calculate the initial population from the proportions and total population size
initial_population = np.round(proportions * population_size).astype(int)

if uploaded_file is not None:
    # Read the CSV into a DataFrame and convert to a NumPy array
    try:
        transition_matrix_df = pd.read_csv(uploaded_file, header=None)
        transition_matrix = transition_matrix_df.values
        num_states = transition_matrix.shape[0]
    except Exception as e:
        st.error(f"Se produjo un error al leer el archivo CSV: {e}")
        st.stop()

# Display current transition matrix and population
st.write("Matriz de transición actual:")
st.write(transition_matrix)

#endregion

#region Definición de funciones de simulación

# Function to simulate the Markov chain for a population
def simulate_markov_chain_population(num_states, initial_population, transition_matrix, num_iterations):
    current_population = initial_population
    population_history = [current_population]

    for _ in range(num_iterations):
        new_population = np.zeros(num_states, dtype=int)

        for i, state_count in enumerate(current_population):
            cumulative_prob = np.cumsum(transition_matrix[i])

            for _ in range(state_count):
                random_value = np.random.uniform(0, 1)
                next_state_idx = np.searchsorted(cumulative_prob, random_value)
                new_population[next_state_idx] += 1

        population_history.append(new_population)
        current_population = new_population

    return population_history

# Function to simulate the Markov chain for a population (first iteration only)
def simulate_markov_chain_population_first_iter(num_states, initial_population, transition_matrix):
    new_population = np.zeros(num_states, dtype=int)
    for i, state_count in enumerate(initial_population):
        cumulative_prob = np.cumsum(transition_matrix[i])
        for _ in range(state_count):
            random_value = np.random.uniform(0, 1)
            next_state_idx = np.searchsorted(cumulative_prob, random_value)
            new_population[next_state_idx] += 1
    return new_population / np.sum(new_population)

# Function to augment population and record proportions after the first iteration of simulation
def augment_population(num_states, initial_population, transition_matrix, batch_size, epochs):
    population_sizes = [initial_population.sum() + (i * batch_size) for i in range(epochs + 1)]
    proportions_over_time = []

    for new_population_size in population_sizes:
        # Scale the initial population to the new population size
        scaled_population = np.round(initial_population * (new_population_size / initial_population.sum())).astype(int)
        # Recalculate the proportions after the first iteration with the new scaled population
        new_proportions = simulate_markov_chain_population_first_iter(num_states, scaled_population, transition_matrix)
        proportions_over_time.append(new_proportions)

    return population_sizes, proportions_over_time

#endregion

#region Gráficas

# Run the simulation with the given parameters
simulated_population_history = simulate_markov_chain_population(num_states, initial_population, transition_matrix, num_iterations)

# Calculate the proportion vector for each iteration
proportion_vectors = [pop / population_size for pop in simulated_population_history]

# Columns for layout
col1, col2 = st.columns((1, 1))

with col1:
    # Plot the proportion vector for each iteration using Plotly
    fig = go.Figure()
    for i in range(num_states):
        fig.add_trace(go.Scatter(
            x=list(range(num_iterations + 1)),
            y=[pv[i] for pv in proportion_vectors],
            mode='lines',
            name=f'Clase {i}'
        ))

    # Update the layout of the plot
    fig.update_layout(
        title=f'Evolución de configuración del sistema de {population_size} objetos en las etapas',
        xaxis_title='Número de iteración',
        yaxis_title='Configuración',
        legend_title='Clases'
    )

    # Display the plot in the Streamlit app
    st.plotly_chart(fig, use_container_width=True)

    st.write("Tabla de configuraciones por iteración")
    df_proportion_vectors = pd.DataFrame(proportion_vectors, columns=[f'Clase {i}' for i in range(num_states)])
    df_proportion_vectors.index.rename("Número de iteración", inplace=True)
    st.dataframe(df_proportion_vectors)    

    st.markdown("""
        ---
                
        **Explicación:**  
                
        En la gráfica superior podemos ver la evolución de las configuraciones en las etapas $k=0,1,2,...,$ considerando una cantidad $N$ de objetos fija.
        Cuando $N$ es muy grande, podemos considerar que los valores que se observan gráficados son los valores que se tendríamos en el proceso de campo medio.  
    """)


with col2:
    batch_size = 100  # Batch size for each population increase
    epochs = 25  # Number of epochs to simulate

    # Calculate augmented populations and their proportions after the first iteration
    population_sizes, proportions_over_time = augment_population(num_states, initial_population, transition_matrix, batch_size, epochs)

    # Plotting the augmented populations and proportions
    augmented_population_fig = go.Figure()
    for i in range(num_states):
        augmented_population_fig.add_trace(go.Scatter(
            x=population_sizes,
            y=[proportion[i] for proportion in proportions_over_time],
            mode='lines',
            name=f'Clase {i}'
        ))

    # Update the layout of the plot
    augmented_population_fig.update_layout(
        title='Convergencia de campo medio (al aumentar población)',
        xaxis_title='Aumento del tamaño de la problación',
        yaxis_title='Proporción de cada clase',
        legend_title='Clases'
    )
    # Display the plot in the Streamlit app
    st.plotly_chart(augmented_population_fig, use_container_width=True)

    st.write("Tabla de convergencia de campo medio con aumento de número de objetos")
    df_proportions_over_time = pd.DataFrame(proportions_over_time, columns=[f'Clase {i}' for i in range(num_states)], index=population_sizes)
    df_proportions_over_time.index.rename("Número de objetos", inplace=True)
    st.dataframe(df_proportions_over_time)    

    st.markdown("""
        ---
        **Explicación:**  
                
        En la gráfica superior podemos ver como ocurre la aproximación hacia el campo medio, si dejamos fija la etapa $k=1$ y aumentamos el valor del número de objetos $N$.
        En este caso, la gráfica considera como valor inicial de $N$ el introducido por el usuario, y va aumentando la cantidad de objetos en bloques de 100. 
    """)


#endregion
