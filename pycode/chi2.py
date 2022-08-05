import numpy as np
import time
import scipy
import math
from VectorMath import Vector
from NumpyHLSL import float3


class ChiSquareTest:
    """
    Implements Pearson's chi-square test for goodness of fit of a distribution
    to a known reference distribution.

    The implementation here specifically compares a Monte Carlo sampling
    strategy on a 2D (or lower dimensional) space against a reference
    distribution obtained by numerically integrating a probability density
    function over grid in the distribution's parameter domain.

    Parameter ``domain`` (object):
       An implementation of the domain interface (``SphericalDomain``, etc.),
       which transforms between the parameter and target domain of the
       distribution

    Parameter ``sample_func`` (function):
       An importance sampling function which maps an array of uniform variates
       of size ``[sample_dim, sample_count]`` to an array of ``sample_count``
       samples on the target domain.

    Parameter ``pdf_func`` (function):
       Function that is expected to specify the probability density of the
       samples produced by ``sample_func``. The test will try to collect
       sufficient statistical evidence to reject this hypothesis.

    Parameter ``sample_dim`` (int):
       Number of random dimensions consumed by ``sample_func`` per sample. The
       default value is ``2``.

    Parameter ``sample_count`` (int):
       Total number of samples to be generated. The test will have more
       evidence as this number tends to infinity. The default value is
       ``1000000``.

    Parameter ``res`` (int):
       Vertical resolution of the generated histograms. The horizontal
       resolution will be calculated as ``res * domain.aspect()``. The
       default value of ``101`` is intentionally an odd number to prevent
       issues with floating point precision at sharp boundaries that may
       separate the domain into two parts (e.g. top hemisphere of a sphere
       parameterization).

    Parameter ``ires`` (int):
       Number of horizontal/vertical subintervals used to numerically integrate
       the probability density over each histogram cell (using the trapezoid
       rule). The default value is ``4``.

    Notes:

    The following attributes are part of the public API:

    messages: string
        The implementation may generate a number of messages while running the
        test, which can be retrieved via this attribute.

    histogram: array
        The histogram array is populated by the ``tabulate_histogram()`` method
        and stored in this attribute.

    pdf: array
        The probability density function array is populated by the
        ``tabulate_pdf()`` method and stored in this attribute.

    p_value: float
        The p-value of the test is computed in the ``run()`` method and stored
        in this attribute.
    """
    def __init__(self, domain, sample_func, pdf_func, sample_dim=2,
                 sample_count=1000000, res=101, ires=8):

        assert res > 0
        assert ires >= 2, "The 'ires' parameter must be >= 2!"

        self.domain = domain
        self.sample_func = sample_func
        self.pdf_func = pdf_func
        self.sample_dim = sample_dim
        self.sample_count = sample_count
        if domain.aspect() == None:
            self.res = np.array([res, 1])
        else:
            self.res = np.maximum(np.array([np.floor(res / domain.aspect()), res]), 1)
        self.ires = ires
        self.bounds = domain.bounds()
        self.pdf = None
        self.histogram = None
        self.p_value = None
        self.messages = ''
        self.fail = False

    def tabulate_histogram(self):
        """
        Invoke the provided sampling strategy many times and generate a
        histogram in the parameter domain. If ``sample_func`` returns a tuple
        ``(positions, weights)`` instead of just positions, the samples are
        considered to be weighted.
        """

        self.histogram_start = time.time()

        # Generate a table of uniform variates
        samples_in = np.random.uniform(size=[self.sample_dim, self.sample_count])

        # Invoke sampling strategy
        samples_out = self.sample_func(samples_in)

        if type(samples_out) is tuple:
            weights_out = samples_out[1]
            samples_out = samples_out[0]
        else:
            weights_out = np.ones(self.sample_count)

        # Map samples into the parameter domain
        xy = self.domain.map_backward(samples_out)

        # Sanity check
        extents = self.bounds[1] - self.bounds[0]
        eps = extents * 1e-4
        in_domain = np.all((xy >= (self.bounds[0] - eps).reshape(-1, 1)) &
                           (xy <= (self.bounds[1] + eps).reshape(-1, 1)))
        if not in_domain:
            self._log('Encountered samples outside of the specified domain')
            self.fail = True

        # Compute a histogram of the positions in the parameter domain
        xy_range = np.transpose(self.bounds)
        self.histogram, _, _ = np.histogram2d(xy[0], xy[1], range=xy_range, bins=self.res.astype(int), weights=weights_out)

        self.histogram_end = time.time()

        self.histogram_sum = np.sum(self.histogram) / self.sample_count
        if self.histogram_sum > 1.01:
            self._log('Sample weights add up to a value greater '
                      'than 1.0: %f' % self.histogram_sum)
            self.fail = True

    def tabulate_pdf(self, simpson_rule=True):
        """
        Numerically integrate the provided probability density function over
        each cell to generate an array resembling the histogram computed by
        ``tabulate_histogram()``. The function uses the trapezoid or simpson rule over
        intervals discretized into ``self.ires`` separate function evaluations.
        """

        self.pdf_start = time.time()

        extents = self.bounds[1] - self.bounds[0]
        endpoint = self.bounds[1] - extents / self.res

        # Compute a set of nodes where the PDF should be evaluated
        y, x = np.meshgrid(
            np.linspace(self.bounds[0][0], endpoint[0], self.res[0].astype(int)),
            np.linspace(self.bounds[0][1], endpoint[1], self.res[1].astype(int))
        )

        endpoint = extents / self.res
        eps = 1e-4
        ny = np.linspace(eps, endpoint[0] * (1 - eps), self.ires)
        nx = np.linspace(eps, endpoint[1] * (1 - eps), self.ires)

        if simpson_rule:
            integral = 0
            vals_y = np.zeros([len(ny), x.shape[0], x.shape[1]])
            vals_x = np.zeros([len(nx), x.shape[0], x.shape[1]])
            samples_y = np.zeros([len(ny), x.shape[0], x.shape[1]])
            samples_x = np.zeros([len(nx), x.shape[0], x.shape[1]])
            for yi, dy in enumerate(ny):
                samples_y[yi] = y + dy

                for xi, dx in enumerate(nx):
                    samples_x[xi] = x + dx

                    p = self.domain.map_forward(np.array([samples_y[yi], samples_x[xi]]))
                    pdf = self.pdf_func(p)

                    vals_x[xi] = pdf
                vals_y[yi] = scipy.integrate.simps(vals_x, samples_x, axis=0)
            integral = scipy.integrate.simps(vals_y, samples_y, axis=0)

            self.pdf = integral * self.sample_count

        else:
            wy = [1 / (self.ires - 1)] * self.ires
            wx = [1 / (self.ires - 1)] * self.ires
            wy[0] = wy[-1] = wy[0] * .5
            wx[0] = wx[-1] = wx[0] * .5

            integral = 0
            for yi, dy in enumerate(ny):
                for xi, dx in enumerate(nx):
                    p = self.domain.map_forward(np.array([y + dy, x + dx]))
                    pdf = self.pdf_func(p)
                    integral += pdf * wx[xi] * wy[yi]

            self.pdf = integral * (np.prod(extents / self.res) * self.sample_count)

        self.pdf = np.transpose(self.pdf)

        self.pdf_end = time.time()

        # A few sanity checks
        pdf_min = np.min(self.pdf) / self.sample_count
        if not pdf_min >= 0:
            self._log('Failure: Encountered a cell with a '
                      'negative PDF value: %f' % pdf_min)
            self.fail = True

        self.pdf_sum = np.sum(self.pdf) / self.sample_count
        if self.pdf_sum > 1.05:
            self._log('Failure: PDF integrates to a value greater '
                      'than 1.0: %f' % self.pdf_sum)
            self.fail = True

    def run(self, significance_level=0.01, test_count=1, quiet=False):
        """
        Run the Chi^2 test

        Parameter ``significance_level`` (float):
            Denotes the desired significance level (e.g. 0.01 for a test at the
            1% significance level)

        Parameter ``test_count`` (int):
            Specifies the total number of statistical tests run by the user.
            This value will be used to adjust the provided significance level
            so that the combination of the entire set of tests has the provided
            significance level.

        Returns → bool:
            ``True`` upon success, ``False`` if the null hypothesis was
            rejected.

        """

        if self.histogram is None:
            self.tabulate_histogram()

        if self.pdf is None:
            self.tabulate_pdf()

        histogram = self.histogram.flatten()
        pdf = self.pdf.flatten()

        index = np.array([i[0] for i in sorted(enumerate(pdf), key=lambda x: x[1])])

        # Sort entries by expected frequency (increasing)
        pdf = pdf[index]
        histogram = histogram[index]

        # Compute chi^2 statistic and pool low-valued cells
        chi2val, dof, pooled_in, pooled_out = self.__chi2(histogram, pdf, 5, pdf.shape[0])

        if dof < 1:
            self._log('Failure: The number of degrees of freedom is too low!')
            self.fail = True

        if np.any(np.equal(pdf, 0) & np.not_equal(histogram, 0)):
            self._log('Failure: Found samples in a cell with expected '
                      'frequency 0. Rejecting the null hypothesis!')
            self.fail = True

        if pooled_in > 0:
            self._log('Pooled %i low-valued cells into %i cells to '
                      'ensure sufficiently high expected cell frequencies'
                      % (pooled_in, pooled_out))

        pdf_time = (self.pdf_end - self.pdf_start) * 1000
        histogram_time = (self.histogram_end - self.histogram_start) * 1000

        self._log('Histogram sum = %f (%.2f ms), PDF sum = %f (%.2f ms)' %
                  (self.histogram_sum, histogram_time, self.pdf_sum, pdf_time))

        self._log('Chi^2 statistic = %f (d.o.f = %i)' % (chi2val, dof))

        # Probability of observing a test statistic at least as
        # extreme as the one here assuming that the distributions match
        # TODO: validation
        self.p_value = 1 - self.__rlgamma(dof / 2, chi2val / 2)

        # Apply the Šidák correction term, since we'll be conducting multiple
        # independent hypothesis tests. This accounts for the fact that the
        # probability of a failure increases quickly when several hypothesis
        # tests are run in sequence.
        significance_level = 1.0 - (1.0 - significance_level) ** (1.0 / test_count)

        if self.fail:
            self._log('Not running the test for reasons listed above. Target '
                      'density and histogram were written to "chi2_data.py')
            result = False
        elif self.p_value < significance_level \
                or not np.isfinite(self.p_value):
            self._log('***** Rejected ***** the null hypothesis (p-value = %f,'
                      ' significance level = %f). Target density and histogram'
                      ' were written to "chi2_data.py".'
                      % (self.p_value, significance_level))
            result = False
        else:
            self._log('Accepted the null hypothesis (p-value = %f, '
                      'significance level = %f)' %
                      (self.p_value, significance_level))
            result = True
        if not quiet:
            print(self.messages)
            if not result:
                self._dump_tables()
        return result


    def __chi2(self, obs, exp, pool_threshold, n):
        """
        brief Compute the Chi^2 statistic and degrees of freedom of the given
        arrays while pooling low-valued entries together

        Given a list of observations counts (``obs[i]``) and expected observation
        counts (``exp[i]``), this function accumulates the Chi^2 statistic, that is,
        ``(obs-exp)^2 / exp`` for each element ``0, ..., n-1``.

        Minimum expected cell frequency. The Chi^2 test statistic is not useful when
        when the expected frequency in a cell is low (e.g. less than 5), because
        normality assumptions break down in this case. Therefore, the implementation
        will merge such low-frequency cells when they fall below the threshold
        specified here. Specifically, low-valued cells with ``exp[i] < pool_threshold``
        are pooled into larger groups that are above the threshold before their
        contents are added to the Chi^2 statistic.

        The function returns the statistic value, degrees of freedom, below-treshold
        entries and resulting number of pooled regions.        
        """
        chsq = 0.0
        pooled_obs = 0.0
        pooled_exp = 0.0
        dof = 0
        n_pooled_in = 0
        n_pooled_out = 0

        for i in range(n):
            if exp[i] == 0 and obs[i] == 0:
                continue

            if exp[i] < pool_threshold:
                pooled_obs += obs[i]
                pooled_exp += exp[i]
                n_pooled_in += 1

                if pooled_exp > pool_threshold:
                    diff = pooled_obs - pooled_exp
                    chsq += (diff*diff) / pooled_exp
                    pooled_obs = pooled_exp = 0;
                    n_pooled_out += 1
                    dof += 1

            else:
                diff = obs[i] - exp[i]
                chsq += (diff*diff) / exp[i]
                dof += 1

        return chsq, dof - 1, n_pooled_in, n_pooled_out


    def __rlgamma(self, a, x):
        'Regularized lower incomplete gamma function based on CEPHES'

        eps = 1e-15
        big = 4.503599627370496e15
        biginv = 2.22044604925031308085e-16

        if a < 0 or x < 0:
            raise "out of range"

        if x == 0:
            return 0

        ax = (a * np.log(x)) - x - math.lgamma(a)

        if ax < -709.78271289338399:
            return 1.0 if a < x else 0.0

        if x <= 1 or x <= a:
            r2 = a
            c2 = 1
            ans2 = 1

            while True:
                r2 = r2 + 1
                c2 = c2 * x / r2
                ans2 += c2

                if c2 / ans2 <= eps:
                    break

            return np.exp(ax) * ans2 / a

        c = 0
        y = 1 - a
        z = x + y + 1
        p3 = 1
        q3 = x
        p2 = x + 1
        q2 = z * x
        ans = p2 / q2

        while True:
            c += 1
            y += 1
            z += 2
            yc = y * c
            p = (p2 * z) - (p3 * yc)
            q = (q2 * z) - (q3 * yc)

            if q != 0:
                nextans = p / q
                error = np.abs((ans - nextans) / nextans)
                ans = nextans
            else:
                error = 1

            p3 = p2
            p2 = p
            q3 = q2
            q2 = q

            # normalize fraction when the numerator becomes large
            if np.abs(p) > big:
                p3 *= biginv
                p2 *= biginv
                q3 *= biginv
                q2 *= biginv

            if error <= eps:
                break;

        return 1 - np.exp(ax) * ans


    def _dump_tables(self):
        with open("chi2_data.py", "w") as f:
            pdf = str([[self.pdf[x + y * self.res.x]
                        for x in range(self.res.x)]
                       for y in range(self.res.y)])
            histogram = str([[self.histogram[x + y * self.res.x]
                              for x in range(self.res.x)]
                             for y in range(self.res.y)])

            f.write("pdf=%s\n" % str(pdf))
            f.write("histogram=%s\n\n" % str(histogram))
            f.write('if __name__ == \'__main__\':\n')
            f.write('    import matplotlib.pyplot as plt\n')
            f.write('    import numpy as np\n\n')
            f.write('    fig, axs = plt.subplots(1,3, figsize=(15, 5))\n')
            f.write('    pdf = np.array(pdf)\n')
            f.write('    histogram = np.array(histogram)\n')
            f.write('    diff=histogram - pdf\n')
            f.write('    absdiff=np.abs(diff).max()\n')
            f.write('    a = pdf.shape[1] / pdf.shape[0]\n')
            f.write('    pdf_plot = axs[0].imshow(pdf, aspect=a,'
                    ' interpolation=\'nearest\')\n')
            f.write('    hist_plot = axs[1].imshow(histogram, aspect=a,'
                    ' interpolation=\'nearest\')\n')
            f.write('    diff_plot = axs[2].imshow(diff, aspect=a, '
                    'vmin=-absdiff, vmax=absdiff, interpolation=\'nearest\','
                    ' cmap=\'coolwarm\')\n')
            f.write('    axs[0].title.set_text(\'PDF\')\n')
            f.write('    axs[1].title.set_text(\'Histogram\')\n')
            f.write('    axs[2].title.set_text(\'Difference\')\n')
            f.write('    props = dict(fraction=0.046, pad=0.04)\n')
            f.write('    fig.colorbar(pdf_plot, ax=axs[0], **props)\n')
            f.write('    fig.colorbar(hist_plot, ax=axs[1], **props)\n')
            f.write('    fig.colorbar(diff_plot, ax=axs[2], **props)\n')
            f.write('    plt.tight_layout()\n')
            f.write('    plt.show()\n')

    def _log(self, msg):
        self.messages += msg + '\n'


# class LineDomain:
#     ' The identity map on the line.'

#     def __init__(self, bounds=[-1.0, 1.0]):
#         from mitsuba.core import ScalarBoundingBox2f

#         self._bounds = ScalarBoundingBox2f(
#             min=(bounds[0], -0.5),
#             max=(bounds[1], 0.5)
#         )

#     def bounds(self):
#         return self._bounds

#     def aspect(self):
#         return None

#     def map_forward(self, p):
#         return p.x

#     def map_backward(self, p):
#         from mitsuba.core import Vector2f, Float
#         return Vector2f(p.x, ek.zero(Float, len(p.x)))


# class PlanarDomain:
#     'The identity map on the plane'

#     def __init__(self, bounds=None):
#         from mitsuba.core import ScalarBoundingBox2f

#         if bounds is None:
#             bounds = ScalarBoundingBox2f(-1, 1)

#         self._bounds = bounds

#     def bounds(self):
#         return self._bounds

#     def aspect(self):
#         extents = self._bounds.extents()
#         return extents.x / extents.y

#     def map_forward(self, p):
#         return p

#     def map_backward(self, p):
#         return p


class SphericalDomain:
    'Maps between the unit sphere and a [cos(theta), phi] parameterization.'

    def bounds(self):
        return np.array([[-1, -np.pi], [1, np.pi]])

    def aspect(self):
        return 2

    def map_forward(self, p):
        cos_theta = p[0]
        sin_theta = np.sqrt(1- cos_theta**2)
        sin_phi, cos_phi = np.sin(p[1]), np.cos(p[1])
        v = float3(cos_phi * sin_theta, sin_phi * sin_theta, cos_theta, keepdims=False)
        return v

    def map_backward(self, p):
        p_v = p.view(Vector)
        s = np.array([p_v.z, np.arctan2(p_v.y, p_v.x)])
        return s


