def deriv(upper, lower, rho):
    return (2 * upper * rho) - (2 * lower * rho)

def find_inertia(m, l):
    dist = (0.05, 0.06, 0.09, 0.125, 0.17, 0.215, 0.18, 0.12) # mass distribution
    n = len(dist)
    cm = 0.42 # center of mass percentile
    rhos = [((n*m)/l) * m_i for m_i in dist] # densities

    sum = 0

    for i in range(1,n+1):
        upper = l*(cm-1+(i/n))
        lower = l*(cm-1+((i-1)/n))
        # print(f'section: {i}')
        # print(lower)
        # print(upper)
        sum += deriv(upper, lower, rhos[i-1])

    print(f'moment of inertia: {sum} cm g^2')

# calculate the moi of a locust with mass m and length l

def main():
    m = 1.5 # mass in grams
    l = 4.5 # length in cm

    find_inertia(m, l)

if __name__ == '__main__':
    main()
