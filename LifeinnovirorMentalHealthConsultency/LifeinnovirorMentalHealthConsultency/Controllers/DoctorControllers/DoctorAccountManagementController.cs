using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Data.Entity;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Threading.Tasks;
using System.Web;
using System.Web.Http;
using System.Xml.Linq;
using LifeinnovirorMentalHealthConsultency.Context;
using LifeinnovirorMentalHealthConsultency.Context.Tables;
using LifeinnovirorMentalHealthConsultency.Functional_Class;
using LifeinnovirorMentalHealthConsultency.Models;
using Newtonsoft.Json;

namespace LifeinnovirorMentalHealthConsultency.Controllers.DoctorControllers
{
    public class DoctorAccountManagementController : ApiController
    {
        private readonly LifeinnovirorContext db;   
        public DoctorAccountManagementController()
        {
            db = new LifeinnovirorContext();
        }



        // It is an async method to handle file and form data
        /*
         * Allows a doctor to submit registration request with optional profile image.
         * Validates input, ensures unique email, processes file if present,
         * sets status as Pending, hashes password and security answer, saves to DB,
         * logs the request, and returns appropriate response.
         */
        [HttpPost]
        [Route("api/doctor/createDoctorAccount")]
        public async Task<IHttpActionResult> CreateDoctor()
        {
            try
            {
                if (!Request.Content.IsMimeMultipartContent())
                {
                    return Content(HttpStatusCode.BadRequest, new
                    {
                        success = false,
                        message = "Expected multipart content with doctor data and optional profile image."
                    });
                }

                //temp path for saving form data
                var tempUploadPath = HttpContext.Current.Server.MapPath(CustomVariables.temporaryFilePath);
                if (!Directory.Exists(tempUploadPath))
                {
                    Directory.CreateDirectory(tempUploadPath);
                }

                var provider = new MultipartFormDataStreamProvider(tempUploadPath);
                await Request.Content.ReadAsMultipartAsync(provider);

                // get data from doctor form
                var doctorJson = provider.FormData["doctor"];
                if (string.IsNullOrWhiteSpace(doctorJson))
                {
                    return Content(HttpStatusCode.BadRequest, new
                    {
                        success = false,
                        message = "Missing doctor data."
                    });
                }

                // map data to Doctor Model
                var model = JsonConvert.DeserializeObject<Doctor>(doctorJson);
                if (model == null)
                {
                    return Content(HttpStatusCode.BadRequest, new
                    {
                        success = false,
                        message = "Invalid doctor data format."
                    });
                }

                // manual validation
                var validationContext = new ValidationContext(model, null, null);
                var validationResults = new List<ValidationResult>();
                bool isValid = Validator.TryValidateObject(model, validationContext, validationResults, true);
                if (!isValid)
                {
                    var errors = validationResults.Select(e => new
                    {
                        Field = string.Join(", ", e.MemberNames),
                        Errors = new List<string> { e.ErrorMessage }
                    });

                    return Content(HttpStatusCode.BadRequest, new
                    {
                        success = false,
                        message = "Validation failed.",
                        errors = errors
                    });
                }

                // Find the latest rejected or pending request
                var lastRejectedRequest = await db.Doctors
                    .Where(d => d.Email == model.Email &&
                               (d.Status == "Rejected" || d.Status == "Pending"))
                    .OrderByDescending(d => d.UpdatedAt)
                    .FirstOrDefaultAsync();

                if (lastRejectedRequest != null && 
                    lastRejectedRequest.UpdatedAt > DateTime.Now.AddDays(-CustomVariables.daysAfterDoctorCanRequestRegistrationAgain))
                {
                    // finding remianing days after he or she can request again
                    var daysRemaining = (lastRejectedRequest.UpdatedAt.AddDays(CustomVariables.daysAfterDoctorCanRequestRegistrationAgain) - DateTime.Now).Days;

                    return Content(HttpStatusCode.Conflict, new
                    {
                        success = false,
                        message = $"You have already requested before. You can request again after {daysRemaining} day(s)."
                    });
                }

                // checking if the doctor is already registered or not
                bool approvedDoctorExists = await db.Doctors.AnyAsync(d => d.Email == model.Email && 
                                                                      d.Status == "Approved");

                if (approvedDoctorExists)
                {
                    return Content(HttpStatusCode.Conflict, new
                    {
                        success = false,
                        message = "An approved doctor with this email already exists."
                    });
                }

                // Set values
                model.PasswordHash = CustomFunctions.GetSha256HashBase64(model.PasswordHash);
                model.SecurityAnswerHash = CustomFunctions.GetSha256HashBase64(model.SecurityAnswerHash);
                model.Status = "Pending";
                model.CreatedAt = DateTime.Now;
                model.UpdatedAt = DateTime.Now;

                //save doctor first to get id
                db.Doctors.Add(model);
                await db.SaveChangesAsync();

                string imageSaveError = null;
                var photo = provider.FileData.FirstOrDefault(); 
                if (photo != null)
                {
                    try
                    {
                        var fileInfo = new FileInfo(photo.LocalFileName);
                        const int maxSize = CustomVariables.maxSizeOfProfilePictureInMB * 1024 * 1024;
                        if (fileInfo.Length > maxSize)
                        {
                            File.Delete(photo.LocalFileName);
                            throw new Exception($"Uploaded image must be less than {CustomVariables.maxSizeOfProfilePictureInMB} MB.");
                        }

                        var allowedExtensions = new[] { ".jpg", ".jpeg", ".png", ".heic" };
                        var extension = Path.GetExtension(photo.Headers.ContentDisposition.FileName.Trim('"')).ToLower();
                        if (!allowedExtensions.Contains(extension))
                        {
                            File.Delete(photo.LocalFileName);
                            throw new Exception("Only .jpg, .jpeg, .png, and .heic files are allowed.");
                        }

                        var photoFolder = HttpContext.Current.Server.MapPath(CustomVariables.doctorProfilePicturesPath);
                        if (!Directory.Exists(photoFolder))
                        {
                            Directory.CreateDirectory(photoFolder);
                        }

                        var finalPath = Path.Combine(photoFolder, model.DoctorId + extension);
                        if (File.Exists(finalPath)) File.Delete(finalPath);

                        File.Move(photo.LocalFileName, finalPath);

                        model.ProfilePhotoUrl = $"{CustomVariables.doctorProfilePicturesPath}/{model.DoctorId}{extension}";
                    }
                    catch (Exception imgEx)
                    {
                        imageSaveError = imgEx.Message;
                    }
                }

                //add success log
                db.SystemLogs.Add(new SystemLog
                {
                    ActorType = "Doctor",
                    ActorId = model.DoctorId,
                    Action = "Doctor Registration Request",
                    Details = $"Doctor '{model.Email}' requested account registration.",
                    CreatedAt = DateTime.Now
                });
                await db.SaveChangesAsync();  //final saving everything

                var message = photo == null ?
                    "Doctor registration request submitted successfully." :
                    imageSaveError == null ?
                        "Doctor registration and profile photo uploaded successfully." :
                        $"Doctor registered, but image upload failed: {imageSaveError}";

                //sending account creation mail
                string mailStatusReport = EmailManagement.AccountCreationMail(model.FullName, model.Email);
                return Ok(new
                {
                    success = true,
                    message = message,
                    data = model,
                    mailStatusReport = mailStatusReport
                });
            }
            catch (Exception ex)
            {
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "Unexpected error during doctor registration.",
                    error = ex.Message
                });
            }
        }




        // It is an async method to handle file and form data
        /*
         * Allows a doctor to submit update request with optional profile image .
         * Validates input, ensures unique email, processes file if present,
         * sets status as Pending, hashes password and security answer, saves to DB,
         * logs the request, and returns appropriate response.
         */
        [Authorize(Roles = "Doctor")]
        [HttpPut]
        [Route("api/doctor/updateDoctorAccount")]
        public async Task<IHttpActionResult> UpdateDoctor()
        {
            try
            {
                if (!Request.Content.IsMimeMultipartContent())
                {
                    return Content(HttpStatusCode.BadRequest, new
                    {
                        success = false,
                        message = "Expected multipart content with doctor data and optional profile image."
                    });
                }

                var tempUploadPath = HttpContext.Current.Server.MapPath(CustomVariables.temporaryFilePath);
                if (!Directory.Exists(tempUploadPath))
                {
                    Directory.CreateDirectory(tempUploadPath);
                }

                var provider = new MultipartFormDataStreamProvider(tempUploadPath);
                await Request.Content.ReadAsMultipartAsync(provider);

                var doctorJson = provider.FormData["doctor"];
                if (string.IsNullOrWhiteSpace(doctorJson))
                {
                    return Content(HttpStatusCode.BadRequest, new
                    {
                        success = false,
                        message = "Missing doctor data."
                    });
                }

                // map data to doctor
                var model = JsonConvert.DeserializeObject<Doctor>(doctorJson);
                if (model == null)
                {
                    return Content(HttpStatusCode.BadRequest, new
                    {
                        success = false,
                        message = "Invalid doctor data format."
                    });
                }

                // manual validation
                var validationContext = new ValidationContext(model, null, null);
                var validationResults = new List<ValidationResult>();
                Validator.TryValidateObject(model, validationContext, validationResults, true);

                // Filter out PasswordHash errors
                var filteredResults = validationResults
                                     .Where(r => !(r.MemberNames.Contains("PasswordHash") ||
                                                   r.MemberNames.Contains("SecurityAnswerHash")))
                                     .ToList();

                // if data is invalid then send invalid sms
                if (filteredResults.Any())
                {
                    var errors = filteredResults.Select(e => new
                    {
                        Field = string.Join(", ", e.MemberNames),
                        Errors = new List<string> { e.ErrorMessage }
                    });

                    return Content(HttpStatusCode.BadRequest, new
                    {
                        success = false,
                        message = "Validation failed.",
                        errors = errors
                    });
                }

                //cehcking if a doctor trying to change another doctor profile or not
                if (model.DoctorId != CustomFunctions.GetDoctorUserIdFromToken(User))
                {
                    return Content(HttpStatusCode.Forbidden, new
                    {
                        success = false,
                        message = "Unauthorized data manupulation"
                    });
                }

                // Find existing doctor by ID
                var existingDoctor = await db.Doctors.FindAsync(model.DoctorId);
                if (existingDoctor == null)
                {
                    return Content(HttpStatusCode.NotFound, new
                    {
                        success = false,
                        message = "Doctor not found."
                    });
                }

                // Only approved doctors can update their profile
                if (existingDoctor.Status != "Approved")
                {
                    return Content(HttpStatusCode.Forbidden, new
                    {
                        success = false,
                        message = "Only approved doctors can update their profile."
                    });
                }


                // Check for email conflict excluding current object
                bool emailExists = await db.Doctors.AnyAsync(p => p.Email == model.Email &&
                                                             p.DoctorId != model.DoctorId);
                if (emailExists)
                {
                    return Content(HttpStatusCode.Conflict, new
                    {
                        success = false,
                        message = "Another account with this email already exists."
                    });
                }

                if (model.SecurityAnswerHash != null)
                {
                    existingDoctor.SecurityAnswerHash = CustomFunctions.GetSha256HashBase64(model.SecurityAnswerHash);
                }

                // Update fields except CreatedAt, Status 
                existingDoctor.Email = model.Email;
                existingDoctor.FullName = model.FullName;
                existingDoctor.PhoneNumber = model.PhoneNumber;
                existingDoctor.SecurityQuestion = model.SecurityQuestion;
                existingDoctor.Qualifications = model.Qualifications;
                existingDoctor.ExperienceSummary = model.ExperienceSummary;
                existingDoctor.YearsOfExperience = model.YearsOfExperience;
                existingDoctor.minimumCancelTime = model.minimumCancelTime;
                existingDoctor.UpdatedAt = DateTime.Now;

                // Handle optional profile photo update
                string imageSaveError = null;
                var photo = provider.FileData.FirstOrDefault();
                if (photo != null)
                {
                    try
                    {
                        var fileInfo = new FileInfo(photo.LocalFileName);
                        const int maxSize = CustomVariables.maxSizeOfProfilePictureInMB * 1024 * 1024;
                        if (fileInfo.Length > maxSize)
                        {
                            File.Delete(photo.LocalFileName);
                            throw new Exception($"Uploaded image must be less than {CustomVariables.maxSizeOfProfilePictureInMB} MB.");
                        }

                        var allowedExtensions = new[] { ".jpg", ".jpeg", ".png", ".heic" };
                        var extension = Path.GetExtension(photo.Headers.ContentDisposition.FileName.Trim('"')).ToLower();
                        if (!allowedExtensions.Contains(extension))
                        {
                            File.Delete(photo.LocalFileName);
                            throw new Exception("Only .jpg, .jpeg, .png, and .heic files are allowed.");
                        }

                        var photoFolder = HttpContext.Current.Server.MapPath(CustomVariables.doctorProfilePicturesPath);
                        if (!Directory.Exists(photoFolder))
                        {
                            Directory.CreateDirectory(photoFolder);
                        }
                        var finalPath = Path.Combine(photoFolder, existingDoctor.DoctorId + extension);
                        if (File.Exists(finalPath))
                        {
                            File.Delete(finalPath);
                        }
                        File.Move(photo.LocalFileName, finalPath);
                        existingDoctor.ProfilePhotoUrl = $"{CustomVariables.doctorProfilePicturesPath}/{existingDoctor.DoctorId}{extension}";
                    }
                    catch (Exception imgEx)
                    {
                        imageSaveError = imgEx.Message;
                    }
                }

                // Log the update action
                db.SystemLogs.Add(new SystemLog
                {
                    ActorType = "Doctor",
                    ActorId = existingDoctor.DoctorId,
                    Action = "Doctor Profile Update",
                    Details = $"Doctor '{existingDoctor.Email}' updated their profile.",
                    CreatedAt = DateTime.Now
                });

                await db.SaveChangesAsync();

                var message = photo == null ? "Doctor profile updated successfully." :
                             imageSaveError == null ? "Doctor profile and photo updated successfully." :
                             $"Doctor updated, but image upload failed: {imageSaveError}";

                return Ok(new
                {
                    success = true,
                    message = message,
                    data = existingDoctor
                });
            }
            catch (Exception ex)
            {
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "Unexpected error during doctor update.",
                    error = ex.Message
                });
            }
        }



        // this will get current logged in doctor
        [Authorize(Roles = "Doctor")]
        [HttpGet]
        [Route("api/doctor/getDoctorAccount")]
        public async Task <IHttpActionResult> GetDoctor()
        {
            try
            {
                var doctor = await db.Doctors.FindAsync(CustomFunctions.GetDoctorUserIdFromToken(User));

                if (doctor == null)
                {
                    return Content(HttpStatusCode.NotFound, new
                    {
                        success = false,
                        message = "Doctor not found."
                    });
                }

                // Map profile photo path
                string photoPath = HttpContext.Current.Server.MapPath(doctor.ProfilePhotoUrl ?? "");
                string base64Image = null;

                if (!string.IsNullOrEmpty(doctor.ProfilePhotoUrl) && File.Exists(photoPath))
                {
                    byte[] imageBytes = File.ReadAllBytes(photoPath);
                    base64Image = Convert.ToBase64String(imageBytes);
                }

                return Ok(new
                {
                    success = true,
                    message = "Doctor retrieved successfully.",
                    data = new
                    {
                        doctor.DoctorId,
                        doctor.FullName,
                        doctor.Email,
                        doctor.PhoneNumber,
                        doctor.Qualifications,
                        doctor.ExperienceSummary,
                        doctor.YearsOfExperience,
                        doctor.minimumCancelTime,
                        doctor.Status,
                        doctor.CreatedAt,
                        doctor.UpdatedAt,
                        doctor.SecurityQuestion,
                        ProfilePhotoBase64 = base64Image
                    }
                });
            }
            catch (Exception ex)
            {
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "An error occurred while retrieving the doctor.",
                    error = ex.Message
                });
            }
        }




        // this will delete current logged doctor
        [Authorize(Roles = "Doctor")]
        [HttpDelete]
        [Route("api/doctor/deleteDoctorAccount")]
        public async Task<IHttpActionResult> DeleteAccount()
        {
            try
            {
                var doctorId = CustomFunctions.GetDoctorUserIdFromToken(User); 

                var doctor = await db.Doctors.FindAsync(doctorId);
                if (doctor == null)
                {
                    return Content(HttpStatusCode.NotFound, new
                    {
                        success = false,
                        message = "Doctor not found."
                    });
                }

                // Delete profile image if exists
                var photoFolder = HttpContext.Current.Server.MapPath(CustomVariables.doctorProfilePicturesPath);
                var existingFiles = Directory.GetFiles(photoFolder, $"{doctorId}.*");
                foreach (var file in existingFiles)
                {
                    File.Delete(file);
                }

                // Delete doctor record
                db.Doctors.Remove(doctor);

                // Log: deletion
                db.SystemLogs.Add(new SystemLog
                {
                    ActorType = "Doctor",
                    ActorId = doctorId,
                    Action = "Delete Doctor",
                    Details = $"Doctor '{doctor.Email}' deleted their account.",
                    CreatedAt = DateTime.Now
                });
                await db.SaveChangesAsync();

                return Ok(new
                {
                    success = true,
                    message = "Account deleted successfully."
                });
            }
            catch (Exception ex)
            {
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "An unexpected error occurred while deleting the doctor account.",
                    error = ex.Message
                });
            }
        }





        [Authorize(Roles = "Doctor")]
        [HttpPost]
        [Route("api/Doctor/changePassword")]
        public async Task<IHttpActionResult> ChangePassword(ChangePasswordModel model)
        {
            try
            {
                // Model validation
                if (!ModelState.IsValid)
                {
                    var errors = ModelState.Where(ms => ms.Value.Errors.Any())
                        .Select(ms => new
                        {
                            Field = ms.Key,
                            Errors = ms.Value.Errors.Select(e => e.ErrorMessage).ToList()
                        });

                    return Content(HttpStatusCode.BadRequest, new
                    {
                        success = false,
                        message = "Validation failed.",
                        errors = errors
                    });
                }


                // Verify if current user exists or not 
                var doctor = await db.Doctors.FindAsync(CustomFunctions.GetDoctorUserIdFromToken(User));
                if (doctor == null)
                {
                    return Content(HttpStatusCode.NotFound, new
                    {
                        success = false,
                        message = "Doctor not found."
                    });
                }

                //verify current password
                if (doctor.PasswordHash != CustomFunctions.GetSha256HashBase64(model.CurrentPassword))
                {
                    return Content(HttpStatusCode.BadRequest, new
                    {
                        success = false,
                        message = "Current password is incorrect."
                    });
                }

                // Update password
                doctor.PasswordHash = CustomFunctions.GetSha256HashBase64(model.NewPassword);

                // Log addition
                db.SystemLogs.Add(new SystemLog
                {
                    ActorType = "Doctor",
                    ActorId = doctor.DoctorId,
                    Action = "Change Password",
                    Details = $"Doctor '{doctor.Email}' changed their account password.",
                    CreatedAt = DateTime.Now
                });
                await db.SaveChangesAsync();

                return Ok(new
                {
                    success = true,
                    message = "Password changed successfully."
                });
            }
            catch (Exception ex)
            {
                // Unexpected server error
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "An unexpected error occurred while changing password.",
                    error = ex.Message
                });
            }

        }

    }
}
